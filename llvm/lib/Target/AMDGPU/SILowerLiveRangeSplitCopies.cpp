//===-- SILowerLiveRangeSplitCopies.cpp - Lower liverange split copies after
// regalloc
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower the LR_SPLIT_COPY instructions inserted for various register classes.
/// AMDGPU target generates LR_SPLIT_COPY instruction for the liverange split
/// copies to distinguish them from the other COPY instructions and this
/// distinction is needed during the regalloc. This pass is inserted after the
/// regalloc pipeline and it lowers LR_SPLIT_COPY into COPY opcode. In the
/// process, it also inserts necessary exec mask manipulation for the
/// whole-wave-copy operations.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-lower-liverange-split-copies"

namespace {

class SILowerLiveRangeSplitCopies : public MachineFunctionPass {
public:
  static char ID;

  SILowerLiveRangeSplitCopies() : MachineFunctionPass(ID) {
    initializeSILowerLiveRangeSplitCopiesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Lower WWM Copies"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool isWWMCopy(const MachineInstr &MI);
  bool isSCCLiveAtMI(const MachineInstr &MI);
  void addToWWMSpills(MachineFunction &MF, Register Reg);
  void lowerBundleCopies(MachineInstr &MI);

  LiveIntervals *LIS;
  SlotIndexes *Indexes;
  VirtRegMap *VRM;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  const MachineRegisterInfo *MRI;
  SIMachineFunctionInfo *MFI;
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SILowerLiveRangeSplitCopies, DEBUG_TYPE,
                      "SI Lower Liverange Split Copies", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(SILowerLiveRangeSplitCopies, DEBUG_TYPE,
                    "SI Lower Liverange Split Copies", false, false)

char SILowerLiveRangeSplitCopies::ID = 0;

char &llvm::SILowerLiveRangeSplitCopiesID = SILowerLiveRangeSplitCopies::ID;

// Returns true if \p MI is a whole-wave copy instruction. Iterate
// recursively skipping the intermediate copies if it maps to any
// whole-wave operation.
bool SILowerLiveRangeSplitCopies::isWWMCopy(const MachineInstr &MI) {
  // Skip if it is a subreg copy.
  if (!TII->isFullCopyInstr(MI))
    return false;

  Register SrcReg = MI.getOperand(1).getReg();

  if (MFI->checkFlag(SrcReg, AMDGPU::VirtRegFlag::WWM_REG))
    return true;

  if (SrcReg.isPhysical())
    return false;

  // Look recursively skipping intermediate copies.
  const MachineInstr *DefMI = MRI->getUniqueVRegDef(SrcReg);
  if (!DefMI || DefMI->getOpcode() != AMDGPU::LR_SPLIT_COPY)
    return false;

  return isWWMCopy(*DefMI);
}

bool SILowerLiveRangeSplitCopies::isSCCLiveAtMI(const MachineInstr &MI) {
  // We can't determine the liveness info if LIS isn't available. Early return
  // in that case and always assume SCC is live.
  if (!LIS)
    return true;

  LiveRange &LR =
      LIS->getRegUnit(*MCRegUnitIterator(MCRegister::from(AMDGPU::SCC), TRI));
  SlotIndex Idx = LIS->getInstructionIndex(MI);
  return LR.liveAt(Idx);
}

// If \p Reg is assigned with a physical VGPR, add the latter into wwm-spills
// for preserving its entire lanes at function prolog/epilog.
void SILowerLiveRangeSplitCopies::addToWWMSpills(MachineFunction &MF,
                                                 Register Reg) {
  if (Reg.isPhysical())
    return;

  Register PhysReg = VRM->getPhys(Reg);
  assert(PhysReg != VirtRegMap::NO_PHYS_REG &&
         "should have allocated a physical register");

  MFI->allocateWWMSpill(MF, PhysReg);
}

void SILowerLiveRangeSplitCopies::lowerBundleCopies(MachineInstr &MI) {
  MachineBasicBlock::instr_iterator I = MI.getIterator();
  while (I->isBundled() && I->getOpcode() == AMDGPU::LR_SPLIT_COPY) {
    I->setDesc(TII->get(AMDGPU::COPY));
    I++;
  }
}
bool SILowerLiveRangeSplitCopies::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  MFI = MF.getInfo<SIMachineFunctionInfo>();
  LIS = getAnalysisIfAvailable<LiveIntervals>();
  Indexes = getAnalysisIfAvailable<SlotIndexes>();
  VRM = getAnalysisIfAvailable<VirtRegMap>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AMDGPU::LR_SPLIT_COPY)
        continue;

      // TODO: Club adjacent WWM ops between same exec save/restore
      if (TII->isVGPRCopy(MI) &&
          !TRI->isSGPRReg(*MRI, MI.getOperand(1).getReg()) &&
          MI.getOperand(0).getReg().isVirtual() && isWWMCopy(MI)) {
        // For WWM vector copies, manipulate the exec mask around the copy
        // instruction.
        DebugLoc DL = MI.getDebugLoc();
        MachineBasicBlock::iterator InsertPt = MI.getIterator();
        Register RegForExecCopy = MFI->getSGPRForEXECCopy();
        TII->insertScratchExecCopy(MF, MBB, InsertPt, DL, RegForExecCopy,
                                   isSCCLiveAtMI(MI), Indexes);
        TII->restoreExec(MF, MBB, ++InsertPt, DL, RegForExecCopy, Indexes);
        addToWWMSpills(MF, MI.getOperand(0).getReg());
        LLVM_DEBUG(dbgs() << "WWM copy manipulation for " << MI);
      }

      // Lower LR_SPLIT_COPY back to COPY
      if (!MI.isBundledWithPred() && MI.isBundledWithSucc())
        lowerBundleCopies(MI);
      else
        MI.setDesc(TII->get(AMDGPU::COPY));
      Changed |= true;
    }
  }

  return Changed;
}
