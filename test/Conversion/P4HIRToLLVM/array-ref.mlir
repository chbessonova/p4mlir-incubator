// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s --implicit-check-not=llvm.select

// A reference to an array element at a constant index is the address of the
// element, which further references can project from.

// CHECK-LABEL: @elements(
// CHECK-DAG:     %[[ARR:.*]] = builtin.unrealized_conversion_cast %arg0 : !{{.*}} to !llvm.ptr
// CHECK-DAG:     %[[V:.*]] = builtin.unrealized_conversion_cast %arg1 : !{{.*}} to i8
// CHECK:         %[[E:.*]] = llvm.getelementptr inbounds %[[ARR]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x struct<packed (i8, i1)>>
// CHECK-NEXT:    %[[X:.*]] = llvm.getelementptr inbounds %[[E]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    llvm.store %[[V]], %[[X]] {alignment = 1 : i64} : i8, !llvm.ptr
// CHECK-NEXT:    %[[R:.*]] = llvm.load %[[ARR]] : !llvm.ptr -> !llvm.array<3 x struct<packed (i8, i1)>>

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !b8i, y: !p4hir.bool>
!arr = !p4hir.array<3 x !s>
#inout = #p4hir<dir inout>

module {
  p4hir.func @elements(%arr : !p4hir.ref<!arr> {p4hir.dir = #inout, p4hir.param_name = "arr"},
                       %v : !b8i) -> !arr {
    %c2 = p4hir.const #p4hir.int<2> : !b32i
    %e = p4hir.array_element_ref %arr[%c2] : !p4hir.ref<!arr>, !b32i
    %x = p4hir.struct_field_ref %e["x"] : <!s>
    p4hir.assign %v, %x : <!b8i>
    %r = p4hir.read %arr : <!arr>
    p4hir.return %r : !arr
  }
}
// -----

// Each memory access is guarded, including accesses through a projected struct field.

// CHECK-LABEL: @write(
// CHECK-NOT:     llvm.alloca
// CHECK-DAG:     %[[I:.*]] = builtin.unrealized_conversion_cast %arg1 : !{{.*}} to i32
// CHECK-DAG:     %[[ARR:.*]] = builtin.unrealized_conversion_cast %arg0 : !{{.*}} to !llvm.ptr
// CHECK:         %[[IDX:.*]] = llvm.zext %[[I]] : i32 to i64
// CHECK-NEXT:    %[[ELT:.*]] = llvm.getelementptr inbounds %[[ARR]][0, %[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x struct<packed (i8, i1)>>
// CHECK-NEXT:    %[[X:.*]] = llvm.getelementptr inbounds %[[ELT]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i8, i1)>
// CHECK-NEXT:    %[[ACCESS_IDX:.*]] = llvm.zext %[[I]] : i32 to i64
// CHECK-NEXT:    %[[SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[VALID:.*]] = llvm.icmp "ult" %[[ACCESS_IDX]], %[[SIZE]] : i64
// CHECK-NEXT:    llvm.cond_br %[[VALID]], ^[[STORE:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
// CHECK-NEXT:  ^[[STORE]]:
// CHECK-NEXT:    llvm.store %{{.*}}, %[[X]] {alignment = 1 : i64} : i8, !llvm.ptr
// CHECK-NEXT:    llvm.br ^[[CONT]]
// CHECK-NEXT:  ^[[CONT]]:
// CHECK-NEXT:    p4hir.return

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !b8i, y: !p4hir.bool>
!arr = !p4hir.array<4 x !s>

module {
  p4hir.func @write(%arr : !p4hir.ref<!arr>, %i : !b32i, %v : !b8i) {
    %e = p4hir.array_element_ref %arr[%i] : !p4hir.ref<!arr>, !b32i
    %x = p4hir.struct_field_ref %e["x"] : <!s>
    p4hir.assign %v, %x : <!b8i>
    p4hir.return
  }
}
// -----

// Signed indices are sign-extended, so that negative ones are out of bounds. The
// GEP may use a truncated index, but the guard uses the original width.
// Constant indices out of bounds are no different from dynamic ones, and arrays of
// size 0 have no element to refer to.

// CHECK-LABEL: @signed(
// CHECK:         %[[IDX:.*]] = llvm.sext %{{.*}} : i8 to i64
// CHECK-NEXT:    llvm.getelementptr inbounds %{{.*}}[0, %[[IDX]]]
// CHECK-NEXT:    %[[ACCESS_IDX:.*]] = llvm.sext %{{.*}} : i8 to i64
// CHECK-NEXT:    %[[SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    llvm.icmp "ult" %[[ACCESS_IDX]], %[[SIZE]] : i64

// CHECK-LABEL: @wide(
// CHECK-DAG:     %[[I:.*]] = builtin.unrealized_conversion_cast %arg1 : !{{.*}} to i128
// CHECK:         %[[TRUNC:.*]] = llvm.trunc %[[I]] : i128 to i64
// CHECK-NEXT:    llvm.getelementptr inbounds %{{.*}}[0, %[[TRUNC]]]
// CHECK-NEXT:    %[[SIZE:.*]] = llvm.mlir.constant(4 : i128) : i128
// CHECK-NEXT:    %[[IN:.*]] = llvm.icmp "ult" %[[I]], %[[SIZE]] : i128
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.zero : i8
// CHECK-NEXT:    llvm.cond_br %[[IN]], ^{{bb[0-9]+}}, ^{{bb[0-9]+}}(%[[ZERO]] : i8)

// CHECK-LABEL: @out_of_bounds(
// CHECK:         %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT:    %[[IDX:.*]] = llvm.zext %[[C7]] : i32 to i64
// CHECK-NEXT:    %[[PTR:.*]] = llvm.getelementptr inbounds %{{.*}}[0, %[[IDX]]]
// CHECK-NEXT:    %[[ACCESS_IDX:.*]] = llvm.zext %[[C7]] : i32 to i64
// CHECK-NEXT:    %[[SIZE:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[VALID:.*]] = llvm.icmp "ult" %[[ACCESS_IDX]], %[[SIZE]] : i64
// CHECK-NEXT:    llvm.cond_br %[[VALID]], ^[[STORE:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
// CHECK-NEXT:    ^[[STORE]]:
// CHECK-NEXT:    llvm.store %{{.*}}, %[[PTR]]
// CHECK-NEXT:    llvm.br ^[[CONT]]
// CHECK-NEXT:    ^[[CONT]]:
// CHECK-NEXT:    p4hir.return

// CHECK-LABEL: @empty(
// CHECK:         llvm.getelementptr inbounds %{{.*}}[0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<0 x i8>
// CHECK:         %[[SIZE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    llvm.icmp "ult" %{{.*}}, %[[SIZE]] : i64

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!b128i = !p4hir.bit<128>
!i8i = !p4hir.int<8>
!arr = !p4hir.array<4 x !b8i>
!empty = !p4hir.array<0 x !b8i>

module {
  p4hir.func @signed(%arr : !p4hir.ref<!arr>, %i : !i8i) -> !b8i {
    %e = p4hir.array_element_ref %arr[%i] : !p4hir.ref<!arr>, !i8i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }

  p4hir.func @wide(%arr : !p4hir.ref<!arr>, %i : !b128i) -> !b8i {
    %e = p4hir.array_element_ref %arr[%i] : !p4hir.ref<!arr>, !b128i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }

  p4hir.func @out_of_bounds(%arr : !p4hir.ref<!arr>, %v : !b8i) {
    %c7 = p4hir.const #p4hir.int<7> : !b32i
    %e = p4hir.array_element_ref %arr[%c7] : !p4hir.ref<!arr>, !b32i
    p4hir.assign %v, %e : <!b8i>
    p4hir.return
  }

  p4hir.func @empty(%arr : !p4hir.ref<!empty>, %i : !b32i) -> !b8i {
    %e = p4hir.array_element_ref %arr[%i] : !p4hir.ref<!empty>, !b32i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}
// -----

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
// -----

// A valid constant beyond the signed 29-bit GEP immediate range uses an SSA
// index, but still needs no guard on its access.
// CHECK-LABEL: @large_constant(
// CHECK-NOT: llvm.icmp
// CHECK: %[[INDEX:.*]] = llvm.mlir.constant(268435456 : i64) : i64
// CHECK-NEXT: llvm.getelementptr inbounds %{{.*}}[0, %[[INDEX]]] : (!llvm.ptr, i64)
// CHECK-NOT: llvm.cond_br
// CHECK: llvm.load
!b8i = !p4hir.bit<8>
!b128i = !p4hir.bit<128>
!arr = !p4hir.array<268435457 x !b8i>
module {
  p4hir.func @large_constant(%a: !p4hir.ref<!arr>) -> !b8i {
    %i = p4hir.const #p4hir.int<268435456> : !b128i
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b128i
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}
