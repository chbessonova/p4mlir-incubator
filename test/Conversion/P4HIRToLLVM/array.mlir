// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Arrays lower to LLVM arrays, arrays of structs and structs of arrays included.
// P4 accepts arrays of size 0, and so does LLVM.

// CHECK-LABEL: @arrays(
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<4 x i8>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<2 x i1>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<3 x struct<packed (i8, array<2 x i16>)>>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<2 x array<3 x i8>>
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<0 x i8>

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!s = !p4hir.struct<"s", x: !b8i, a: !p4hir.array<2 x !b16i>>

module {
  p4hir.func @arrays() {
    %bits = p4hir.variable ["bits", init] : <!p4hir.array<4 x !b8i>>
    %bools = p4hir.variable ["bools", init] : <!p4hir.array<2 x !p4hir.bool>>
    %structs = p4hir.variable ["structs", init] : <!p4hir.array<3 x !s>>
    %nested = p4hir.variable ["nested", init] : <!p4hir.array<2 x !p4hir.array<3 x !b8i>>>
    %empty = p4hir.variable ["empty", init] : <!p4hir.array<0 x !b8i>>
    p4hir.return
  }
}

// -----

// An array is built by inserting its elements in order into a poison value, and an
// element at a constant index is read with `llvm.extractvalue`, whatever the width
// and signedness of the index.

// CHECK-LABEL: @get(
// CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : !b8i to i8
// CHECK-DAG:     %[[B:.*]] = builtin.unrealized_conversion_cast %arg1 : !b8i to i8
// CHECK:         %[[P:.*]] = llvm.mlir.poison : !llvm.array<2 x i8>
// CHECK-NEXT:    %[[A0:.*]] = llvm.insertvalue %[[A]], %[[P]][0] : !llvm.array<2 x i8>
// CHECK-NEXT:    %[[A1:.*]] = llvm.insertvalue %[[B]], %[[A0]][1] : !llvm.array<2 x i8>
// CHECK:         %[[E0:.*]] = llvm.extractvalue %[[A1]][1] : !llvm.array<2 x i8>
// CHECK:         %[[E1:.*]] = llvm.extractvalue %[[A1]][0] : !llvm.array<2 x i8>
// CHECK-NEXT:    llvm.add %[[E0]], %[[E1]] : i8

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!i8i = !p4hir.int<8>
!arr = !p4hir.array<2 x !b8i>

module {
  p4hir.func @get(%a : !b8i, %b : !b8i) -> !b8i {
    %arr = p4hir.array [%a, %b] : !arr
    %c1 = p4hir.const #p4hir.int<1> : !b32i
    %e0 = p4hir.array_get %arr[%c1] : !arr, !b32i
    %c0 = p4hir.const #p4hir.int<0> : !i8i
    %e1 = p4hir.array_get %arr[%c0] : !arr, !i8i
    %s = p4hir.binop(add, %e0, %e1) : !b8i
    p4hir.return %s : !b8i
  }
}

// -----

// Arrays of structs and structs of arrays compose.

// CHECK-LABEL: @nested(
// CHECK-NEXT:    %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : !{{.*}} to !llvm.array<3 x struct<packed (i8, array<2 x i16>)>>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[S:.*]] = llvm.extractvalue %[[A]][2] : !llvm.array<3 x struct<packed (i8, array<2 x i16>)>>
// CHECK-NEXT:    %[[B:.*]] = llvm.extractvalue %[[S]][1] : !llvm.struct<packed (i8, array<2 x i16>)>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[E:.*]] = llvm.extractvalue %[[B]][1] : !llvm.array<2 x i16>
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[E]] : i16 to !b16i

!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !b8i, a: !p4hir.array<2 x !b16i>>
!arr = !p4hir.array<3 x !s>

module {
  p4hir.func @nested(%arr : !arr) -> !b16i {
    %c2 = p4hir.const #p4hir.int<2> : !b32i
    %s = p4hir.array_get %arr[%c2] : !arr, !b32i
    %a = p4hir.struct_extract %s["a"] : !s
    %c1 = p4hir.const #p4hir.int<1> : !b32i
    %e = p4hir.array_get %a[%c1] : !p4hir.array<2 x !b16i>, !b32i
    p4hir.return %e : !b16i
  }
}

// -----

// Array constants are built from their elements, unless they are all zeros.

// CHECK-LABEL: @constants(
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:    %[[P:.*]] = llvm.mlir.poison : !llvm.array<2 x i8>
// CHECK-NEXT:    %[[A0:.*]] = llvm.insertvalue %[[ONE]], %[[P]][0] : !llvm.array<2 x i8>
// CHECK-NEXT:    %[[A1:.*]] = llvm.insertvalue %[[ZERO]], %[[A0]][1] : !llvm.array<2 x i8>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.zero : !llvm.array<2 x i8>
// CHECK-NEXT:    %{{.*}} = llvm.mlir.zero : !llvm.array<0 x i8>

!b8i = !p4hir.bit<8>
#int0_b8i = #p4hir.int<0> : !b8i
#int1_b8i = #p4hir.int<1> : !b8i

module {
  p4hir.func @constants() {
    %some = p4hir.const #p4hir.aggregate<[#int1_b8i, #int0_b8i]> : !p4hir.array<2 x !b8i>
    %zero = p4hir.const #p4hir.aggregate<[#int0_b8i, #int0_b8i]> : !p4hir.array<2 x !b8i>
    %empty = p4hir.const #p4hir.aggregate<[]> : !p4hir.array<0 x !b8i>
    p4hir.return
  }
}

// -----

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

// Arrays are equal if all their elements are, arrays of structs included.

// CHECK-LABEL: @eq(
// CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : !{{.*}} to !llvm.array<2 x struct<packed (i8)>>
// CHECK-DAG:     %[[B:.*]] = builtin.unrealized_conversion_cast %arg1 : !{{.*}} to !llvm.array<2 x struct<packed (i8)>>
// CHECK:         %[[A0:.*]] = llvm.extractvalue %[[A]][0] : !llvm.array<2 x struct<packed (i8)>>
// CHECK-NEXT:    %[[B0:.*]] = llvm.extractvalue %[[B]][0] : !llvm.array<2 x struct<packed (i8)>>
// CHECK-NEXT:    %[[A0X:.*]] = llvm.extractvalue %[[A0]][0] : !llvm.struct<packed (i8)>
// CHECK-NEXT:    %[[B0X:.*]] = llvm.extractvalue %[[B0]][0] : !llvm.struct<packed (i8)>
// CHECK-NEXT:    %[[EQ0:.*]] = llvm.icmp "eq" %[[A0X]], %[[B0X]] : i8
// CHECK-NEXT:    %[[A1:.*]] = llvm.extractvalue %[[A]][1] : !llvm.array<2 x struct<packed (i8)>>
// CHECK-NEXT:    %[[B1:.*]] = llvm.extractvalue %[[B]][1] : !llvm.array<2 x struct<packed (i8)>>
// CHECK-NEXT:    %[[A1X:.*]] = llvm.extractvalue %[[A1]][0] : !llvm.struct<packed (i8)>
// CHECK-NEXT:    %[[B1X:.*]] = llvm.extractvalue %[[B1]][0] : !llvm.struct<packed (i8)>
// CHECK-NEXT:    %[[EQ1:.*]] = llvm.icmp "eq" %[[A1X]], %[[B1X]] : i8
// CHECK-NEXT:    %[[EQ:.*]] = llvm.and %[[EQ0]], %[[EQ1]] : i1
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[EQ]] : i1 to !{{.*}}

// CHECK-LABEL: @ne_empty(
// CHECK-NEXT:    %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:    %[[TRUE2:.*]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:    %[[NE:.*]] = llvm.xor %[[TRUE]], %[[TRUE2]] : i1
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[NE]] : i1 to !{{.*}}

!b8i = !p4hir.bit<8>
!s = !p4hir.struct<"s", x: !b8i>
!arr = !p4hir.array<2 x !s>
!empty = !p4hir.array<0 x !b8i>

module {
  p4hir.func @eq(%a : !arr, %b : !arr) -> !p4hir.bool {
    %eq = p4hir.cmp(eq, %a : !arr, %b : !arr)
    p4hir.return %eq : !p4hir.bool
  }

  p4hir.func @ne_empty(%a : !empty, %b : !empty) -> !p4hir.bool {
    %ne = p4hir.cmp(ne, %a : !empty, %b : !empty)
    p4hir.return %ne : !p4hir.bool
  }
}
