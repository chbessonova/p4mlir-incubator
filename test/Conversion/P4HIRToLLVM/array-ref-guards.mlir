// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s --implicit-check-not=llvm.select

// Multiple uses of the same reference must each be guarded.
// CHECK-LABEL: @reuse(
// CHECK: llvm.cond_br %{{.*}}, ^[[WRITE:bb[0-9]+]], ^[[AFTER_WRITE:bb[0-9]+]]
// CHECK-NEXT: ^[[WRITE]]:
// CHECK-NEXT: llvm.store
// CHECK-NEXT: llvm.br ^[[AFTER_WRITE]]
// CHECK-NEXT: ^[[AFTER_WRITE]]:
// CHECK: %[[ZERO:.*]] = llvm.mlir.zero : i16
// CHECK-NEXT: llvm.cond_br %{{.*}}, ^[[READ:bb[0-9]+]], ^[[AFTER_READ:bb[0-9]+]](%[[ZERO]] : i16)
// CHECK-NEXT: ^[[READ]]:
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load
// CHECK-NEXT: llvm.br ^[[AFTER_READ]](%[[VALUE]] : i16)
// CHECK-NEXT: ^[[AFTER_READ]](%[[RESULT:.*]]: i16):
// CHECK: llvm.cond_br
// CHECK: llvm.load
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!arr = !p4hir.array<4 x !b16i>
module {
  p4hir.func @reuse(%a: !p4hir.ref<!arr>, %i: !b32i, %v: !b16i) -> !b16i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    p4hir.assign %v, %e : <!b16i>
    %r = p4hir.read %e : <!b16i>
    %r2 = p4hir.read %e : <!b16i>
    %sum = p4hir.binop(add, %r, %r2) : !b16i
    p4hir.return %sum : !b16i
  }
}

// -----

// Both array bounds matter through a struct field. Packed field alignment must
// survive the guarded access. The guard checks the full outer index.
// CHECK-LABEL: @nested(
// CHECK: %[[I:.*]] = builtin.unrealized_conversion_cast %arg1 : !{{.*}} to i128
// CHECK: llvm.trunc %[[I]] : i128 to i64
// CHECK: llvm.icmp "ult" %[[I]], %{{.*}} : i128
// CHECK: %[[BOTH:.*]] = llvm.and %{{.*}}, %{{.*}} : i1
// CHECK: llvm.cond_br %[[BOTH]], ^[[ACCESS:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
// CHECK-NEXT: ^[[ACCESS]]:
// CHECK-NEXT: llvm.store %{{.*}}, %{{.*}} {alignment = 1 : i64} : i16, !llvm.ptr
// CHECK-NEXT: llvm.br ^[[CONT]]
// CHECK: llvm.and
// CHECK: llvm.cond_br
// CHECK: llvm.load %{{.*}} {alignment = 1 : i64} : !llvm.ptr -> i16
!b16i = !p4hir.bit<16>
!i8i = !p4hir.int<8>
!b128i = !p4hir.bit<128>
!inner = !p4hir.array<3 x !b16i>
!s = !p4hir.struct<"s", pad: !p4hir.bit<8>, a: !inner>
!arr = !p4hir.array<2 x !s>
module {
  p4hir.func @nested(%a: !p4hir.ref<!arr>, %i: !b128i, %j: !i8i, %v: !b16i) -> !b16i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b128i
    %field = p4hir.struct_field_ref %e["a"] : <!s>
    %cell = p4hir.array_element_ref %field[%j] : !p4hir.ref<!inner>, !i8i
    p4hir.assign %v, %cell : <!b16i>
    %r = p4hir.read %cell : <!b16i>
    p4hir.return %r : !b16i
  }
}

// -----

// Empty arrays still guard every access, including with a zero-width index.
// CHECK-LABEL: @empty_zero_width(
// CHECK: llvm.mlir.constant(0 : i64)
// CHECK: llvm.icmp "ult"
// CHECK: llvm.cond_br
// CHECK: llvm.store
// CHECK: %[[ZERO:.*]] = llvm.mlir.zero : i8
// CHECK-NEXT: llvm.cond_br %{{.*}}, ^{{bb[0-9]+}}, ^{{bb[0-9]+}}(%[[ZERO]] : i8)
// CHECK: llvm.load
!b0i = !p4hir.bit<0>
!b8i = !p4hir.bit<8>
!arr = !p4hir.array<0 x !b8i>
module {
  p4hir.func @empty_zero_width(%a: !p4hir.ref<!arr>, %i: !b0i, %v: !b8i) -> !b8i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b0i
    p4hir.assign %v, %e : <!b8i>
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}

// -----

// Guard aggregate loads too: the false edge supplies an LLVM zero aggregate.
// CHECK-LABEL: @aggregate(
// CHECK: %[[ZERO:.*]] = llvm.mlir.zero : !llvm.struct<packed (i8, i16)>
// CHECK-NEXT: llvm.cond_br %{{.*}}, ^[[READ:bb[0-9]+]], ^[[CONT:bb[0-9]+]](%[[ZERO]] : !llvm.struct<packed (i8, i16)>)
// CHECK-NEXT: ^[[READ]]:
// CHECK-NEXT: %[[R:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<packed (i8, i16)>
// CHECK-NEXT: llvm.br ^[[CONT]](%[[R]] : !llvm.struct<packed (i8, i16)>)
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !p4hir.bit<8>, y: !p4hir.bit<16>>
!arr = !p4hir.array<2 x !s>
module {
  p4hir.func @aggregate(%a: !p4hir.ref<!arr>, %i: !b32i) -> !s {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    %r = p4hir.read %e : <!s>
    p4hir.return %r : !s
  }
}

// -----

// A reference defined before a loop is checked at each access in its body. The
// bounds use the captured SSA index, not a later read of an index variable.
// CHECK-LABEL: @loop(
// CHECK: llvm.br ^[[LOOP:bb[0-9]+]]
// CHECK: ^[[LOOP]]:
// CHECK: llvm.cond_br %{{.*}}, ^[[WRITE:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
// CHECK-NEXT: ^[[WRITE]]:
// CHECK-NEXT: llvm.store
// CHECK-NEXT: llvm.br ^[[CONT]]
// CHECK-NEXT: ^[[CONT]]:
// CHECK: llvm.cond_br %{{.*}}, ^[[LOOP]], ^{{bb[0-9]+}}
!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!arr = !p4hir.array<4 x !b8i>
module {
  p4hir.func @loop(%a: !p4hir.ref<!arr>, %i: !b32i, %v: !b8i, %again: !p4hir.bool) {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    p4hir.br ^loop
  ^loop:
    p4hir.assign %v, %e : <!b8i>
    p4hir.cond_br %again ^loop, ^done
  ^done:
    p4hir.return
  }
}

// -----

// A negative constant takes the zero edge of the guarded read.
// CHECK-LABEL: @negative_constant(
// CHECK: %[[NEG:.*]] = llvm.mlir.constant(-1 : i8) : i8
// CHECK: llvm.getelementptr inbounds
// CHECK-NEXT: %[[INDEX:.*]] = llvm.sext %[[NEG]] : i8 to i64
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VALID:.*]] = llvm.icmp "ult" %[[INDEX]], %[[SIZE]] : i64
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.zero : i8
// CHECK-NEXT: llvm.cond_br %[[VALID]], ^{{bb[0-9]+}}, ^{{bb[0-9]+}}(%[[ZERO]] : i8)

// A bit<0> index denotes zero, which is in bounds for a nonempty array.
// CHECK-LABEL: @nonempty_zero_width(
// CHECK: llvm.getelementptr inbounds
// CHECK-NEXT: %[[INDEX:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[VALID:.*]] = llvm.icmp "ult" %[[INDEX]], %[[SIZE]] : i64
// CHECK: llvm.cond_br %[[VALID]], ^[[LOAD:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK-NEXT: ^[[LOAD]]:
// CHECK-NEXT: llvm.load

!b0i = !p4hir.bit<0>
!b8i = !p4hir.bit<8>
!i8i = !p4hir.int<8>
!arr = !p4hir.array<4 x !b8i>
module {
  p4hir.func @negative_constant(%a: !p4hir.ref<!arr>) -> !b8i {
    %i = p4hir.const #p4hir.int<-1> : !i8i
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !i8i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
  p4hir.func @nonempty_zero_width(%a: !p4hir.ref<!arr>, %i: !b0i) -> !b8i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b0i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}
