// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Constant OOB reads choose zero.
// CHECK-LABEL: @oob(
// CHECK: llvm.mlir.zero : !llvm.struct<packed (i8)>
// CHECK: llvm.mlir.zero : !llvm.struct<packed (i8)>
// CHECK: llvm.mlir.zero : !llvm.struct<packed (i8)>
!s = !p4hir.struct<"s", x: !p4hir.bit<8>>
!arr = !p4hir.array<2 x !s>
!empty = !p4hir.array<0 x !s>
module {
  p4hir.func @oob(%a: !arr, %empty: !empty) -> !s {
    %large = p4hir.const #p4hir.int<18446744073709551616> : !p4hir.bit<128>
    %neg = p4hir.const #p4hir.int<-1> : !p4hir.int<8>
    %zero = p4hir.const #p4hir.int<0> : !p4hir.bit<8>
    %x = p4hir.array_get %a[%large] : !arr, !p4hir.bit<128>
    %y = p4hir.array_get %a[%neg] : !arr, !p4hir.int<8>
    %z = p4hir.array_get %empty[%zero] : !empty, !p4hir.bit<8>
    p4hir.return %z : !s
  }
}

// -----

// A dynamic index remains unsupported even for an empty array. A valid position
// beyond LLVM's 32-bit extractvalue encoding must not be silently truncated.
// CHECK-LABEL: @unsupported(
// CHECK: p4hir.array_get
// CHECK: p4hir.array_get
!b8i = !p4hir.bit<8>
!b64i = !p4hir.bit<64>
!large = !p4hir.array<4294967297 x !b8i>
!empty = !p4hir.array<0 x !b8i>
module {
  p4hir.func @unsupported(%a: !large, %empty: !empty, %i: !b64i) -> !b8i {
    %dyn = p4hir.array_get %empty[%i] : !empty, !b64i
    %pos = p4hir.const #p4hir.int<4294967296> : !b64i
    %r = p4hir.array_get %a[%pos] : !large, !b64i
    p4hir.return %r : !b8i
  }
}
