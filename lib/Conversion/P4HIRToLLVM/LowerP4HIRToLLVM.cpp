// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/P4HIRToLLVM/P4HIRToLLVM.h"
#include "p4mlir/Dialect/P4HIR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"  // IWYU pragma: keep (required for Passes.cpp.inc)
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define DEBUG_TYPE "p4hir-to-llvm"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERP4HIRTOLLVM
#include "p4mlir/Conversion/P4HIRToLLVM/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {

template <typename Op>
LogicalResult lowerToOp(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<Op>(op, operands);
    return success();
}

template <typename Op>
LogicalResult lowerToOp(Operation *op, P4HIR::BinOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) {
    return lowerToOp<Op>(op, adaptor.getOperands(), rewriter);
}

// LLVM integers are signless; signedness comes from the original BitsType.
template <typename SignedOp, typename UnsignedOp>
LogicalResult lowerToSignedOrUnsignedOp(P4HIR::BinOp op, P4HIR::BinOp::Adaptor adaptor,
                                        ConversionPatternRewriter &rewriter) {
    if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(op.getType())) {
        if (bitsType.isSigned()) {
            rewriter.replaceOpWithNewOp<SignedOp>(op, adaptor.getOperands());
        } else {
            rewriter.replaceOpWithNewOp<UnsignedOp>(op, adaptor.getOperands());
        }
        return success();
    }
    return rewriter.notifyMatchFailure(op, "expected bits type");
}

template <typename UnsignedOp>
LogicalResult lowerToUnsignedDivisionOp(P4HIR::BinOp op, P4HIR::BinOp::Adaptor adaptor,
                                        ConversionPatternRewriter &rewriter) {
    auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(op.getType());
    if (!bitsType) {
        return rewriter.notifyMatchFailure(op, "expected bits type");
    }
    if (bitsType.isSigned()) {
        return rewriter.notifyMatchFailure(op, "not defined on signed values");
    }
    return lowerToOp<UnsignedOp>(op, adaptor, rewriter);
}

struct BinOpConversion : public ConvertOpToLLVMPattern<P4HIR::BinOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::BinOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (P4HIR::isZeroWidth(op.getType()))
            return rewriter.notifyMatchFailure(op, "expected bit<0> operation to be folded");

        switch (op.getKind()) {
            case P4HIR::BinOpKind::Add:
                return lowerToOp<LLVM::AddOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::AddSat:
                return lowerToSignedOrUnsignedOp<LLVM::SAddSat, LLVM::UAddSat>(op, adaptor,
                                                                               rewriter);
            case P4HIR::BinOpKind::Sub:
                return lowerToOp<LLVM::SubOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::SubSat:
                return lowerToSignedOrUnsignedOp<LLVM::SSubSat, LLVM::USubSat>(op, adaptor,
                                                                               rewriter);
            case P4HIR::BinOpKind::Mul:
                return lowerToOp<LLVM::MulOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Div:
                return lowerToUnsignedDivisionOp<LLVM::UDivOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Mod:
                return lowerToUnsignedDivisionOp<LLVM::URemOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::And:
                return lowerToOp<LLVM::AndOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Or:
                return lowerToOp<LLVM::OrOp>(op, adaptor, rewriter);
            case P4HIR::BinOpKind::Xor:
                return lowerToOp<LLVM::XOrOp>(op, adaptor, rewriter);
        }
        return rewriter.notifyMatchFailure(op, "unsupported binop kind");
    }
};

struct UnaryOpConversion : public ConvertOpToLLVMPattern<P4HIR::UnaryOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::UnaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (P4HIR::isZeroWidth(op.getType()))
            return rewriter.notifyMatchFailure(op, "expected bit<0> operation to be folded");

        auto input = adaptor.getInput();
        auto intType = cast<IntegerType>(input.getType());
        auto width = intType.getWidth();

        auto createConstant = [&](const APInt &value) -> Value {
            return LLVM::ConstantOp::create(rewriter, op.getLoc(), intType, value);
        };

        switch (op.getKind()) {
            case P4HIR::UnaryOpKind::UPlus:
                rewriter.replaceOp(op, input);
                return success();
            case P4HIR::UnaryOpKind::Neg:  // `-x` is emitted as `0 - x`
                return lowerToOp<LLVM::SubOp>(op, {createConstant(APInt::getZero(width)), input},
                                              rewriter);
            case P4HIR::UnaryOpKind::Cmpl:  // `~x` is emitted as `x ^ all-ones`
                return lowerToOp<LLVM::XOrOp>(op, {input, createConstant(APInt::getAllOnes(width))},
                                              rewriter);
            case P4HIR::UnaryOpKind::LNot:  // `!x` is emitted as `x ^ 1`
                return lowerToOp<LLVM::XOrOp>(op, {input, createConstant(APInt(width, 1))},
                                              rewriter);
        }
        return rewriter.notifyMatchFailure(op, "unsupported unary op kind");
    }
};

struct CmpOpConversion : public ConvertOpToLLVMPattern<P4HIR::CmpOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto lhsType = op.getLhs().getType();
        if (!isa<P4HIR::BitsType, P4HIR::BoolType>(lhsType)) {
            return rewriter.notifyMatchFailure(op, "unsupported cmp operand type");
        }
        if (P4HIR::isZeroWidth(lhsType))
            return rewriter.notifyMatchFailure(op, "expected bit<0> operation to be folded");

        bool isSigned = false;  // BoolType lowers to i1 and is always compared as unsigned.
        if (auto bitsType = dyn_cast<P4HIR::BitsType>(lhsType)) isSigned = bitsType.isSigned();

        auto lowerToICmpOp = [&](LLVM::ICmpPredicate predicate) {
            rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, predicate, adaptor.getLhs(),
                                                      adaptor.getRhs());
            return success();
        };

        switch (op.getKind()) {
            case P4HIR::CmpOpKind::Eq:
                return lowerToICmpOp(LLVM::ICmpPredicate::eq);
            case P4HIR::CmpOpKind::Ne:
                return lowerToICmpOp(LLVM::ICmpPredicate::ne);
            case P4HIR::CmpOpKind::Lt:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::slt
                                              : LLVM::ICmpPredicate::ult);
            case P4HIR::CmpOpKind::Le:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::sle
                                              : LLVM::ICmpPredicate::ule);
            case P4HIR::CmpOpKind::Gt:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::sgt
                                              : LLVM::ICmpPredicate::ugt);
            case P4HIR::CmpOpKind::Ge:
                return lowerToICmpOp(isSigned ? LLVM::ICmpPredicate::sge
                                              : LLVM::ICmpPredicate::uge);
        }
        return rewriter.notifyMatchFailure(op, "unsupported cmp op kind");
    }
};

// Converts a successor block's argument types to match the converted branch
// operands. Entry blocks have no predecessors; their signatures are converted
// when lowering the operation that owns the region.
// Follows the upstream ControlFlowToLLVM lowering.
FailureOr<Block *> getConvertedBlock(ConversionPatternRewriter &rewriter,
                                     const TypeConverter *converter, Operation *branchOp,
                                     Block *block, TypeRange expectedTypes) {
    assert(!block->isEntryBlock() && "entry blocks have no predecessors");

    // There is nothing to do if the types already match, e.g. if the block was already converted
    // for another predecessor.
    if (block->getArgumentTypes() == expectedTypes) return block;

    auto conversion = converter->convertBlockSignature(block);
    if (!conversion)
        return rewriter.notifyMatchFailure(branchOp, "could not compute block signature");
    if (expectedTypes != conversion->getConvertedTypes())
        return rewriter.notifyMatchFailure(branchOp,
                                           "block signature does not match branch operands");
    return rewriter.applySignatureConversion(block, *conversion, converter);
}

struct BrOpConversion : public ConvertOpToLLVMPattern<P4HIR::BrOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::BrOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto dest = getConvertedBlock(rewriter, getTypeConverter(), op, op.getDest(),
                                      TypeRange(adaptor.getDestOperands()));
        if (failed(dest)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, adaptor.getDestOperands(), *dest);
        return success();
    }
};

struct CondBrOpConversion : public ConvertOpToLLVMPattern<P4HIR::CondBrOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CondBrOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto destTrue = getConvertedBlock(rewriter, getTypeConverter(), op, op.getDestTrue(),
                                          TypeRange(adaptor.getDestOperandsTrue()));
        if (failed(destTrue)) return failure();
        auto destFalse = getConvertedBlock(rewriter, getTypeConverter(), op, op.getDestFalse(),
                                           TypeRange(adaptor.getDestOperandsFalse()));
        if (failed(destFalse)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, adaptor.getCond(), *destTrue,
                                                    adaptor.getDestOperandsTrue(), *destFalse,
                                                    adaptor.getDestOperandsFalse());
        return success();
    }
};

Value createIntCast(Value value, IntegerType resultType, Location loc, OpBuilder &builder,
                    bool isSigned) {
    auto valueType = cast<IntegerType>(value.getType());
    if (valueType.getWidth() < resultType.getWidth()) {
        if (isSigned) return LLVM::SExtOp::create(builder, loc, resultType, value);
        return LLVM::ZExtOp::create(builder, loc, resultType, value);
    }
    if (valueType.getWidth() > resultType.getWidth())
        return LLVM::TruncOp::create(builder, loc, resultType, value);
    return value;
}

struct ConcatOpConversion : public ConvertOpToLLVMPattern<P4HIR::ConcatOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ConcatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (P4HIR::isZeroWidth(op.getLhs().getType()) || P4HIR::isZeroWidth(op.getRhs().getType()))
            return rewriter.notifyMatchFailure(op, "expected bit<0> operation to be folded");

        auto highType = cast<IntegerType>(adaptor.getLhs().getType());
        auto lowType = cast<IntegerType>(adaptor.getRhs().getType());

        auto loc = op.getLoc();
        auto lowWidth = lowType.getWidth();
        auto resultType = rewriter.getIntegerType(highType.getWidth() + lowWidth);

        // Concatenation operates on the bit patterns of its operands, regardless of
        // their signedness. Zero-extend both operands so that sign extension cannot
        // introduce bits into the other half of the result.
        Value high = createIntCast(adaptor.getLhs(), resultType, loc, rewriter, /*isSigned=*/false);
        Value low = createIntCast(adaptor.getRhs(), resultType, loc, rewriter, /*isSigned=*/false);
        // The shift amount is the width of the low half, which is always smaller
        // than the result width.
        Value shift = LLVM::ConstantOp::create(rewriter, loc, resultType, lowWidth);
        high = LLVM::ShlOp::create(rewriter, loc, high, shift);
        rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, high, low);
        return success();
    }
};

struct SliceOpConversion : public ConvertOpToLLVMPattern<P4HIR::SliceOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::SliceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto inputType = cast<IntegerType>(adaptor.getInput().getType());

        auto loc = op.getLoc();
        auto resultType = rewriter.getIntegerType(op.getHighBit() - op.getLowBit() + 1);

        Value value = adaptor.getInput();
        if (uint32_t lowBit = op.getLowBit(); lowBit > 0) {
            // Slicing operates on the bit pattern, so shift logically regardless of the
            // signedness of the input.
            Value shift = LLVM::ConstantOp::create(rewriter, loc, inputType, lowBit);
            value = LLVM::LShrOp::create(rewriter, loc, value, shift);
        }
        rewriter.replaceOp(op, createIntCast(value, resultType, loc, rewriter, /*isSigned=*/false));
        return success();
    }
};

struct CastOpConversion : public ConvertOpToLLVMPattern<P4HIR::CastOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto src = adaptor.getSrc();
        auto srcType = dyn_cast<IntegerType>(src.getType());
        auto resultType = getTypeConverter()->convertType<IntegerType>(op.getType());
        if (!srcType || !resultType)
            return rewriter.notifyMatchFailure(op, "unsupported cast type");

        if (srcType == resultType) {
            rewriter.replaceOp(op, src);
            return success();
        }

        if (op.isTruncationCast())
            rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultType, src);
        else if (op.isSignExtensionCast())
            rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, resultType, src);
        else if (op.isZeroExtensionCast())
            rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, resultType, src);
        else
            return rewriter.notifyMatchFailure(op, "unsupported cast");
        return success();
    }
};

// Builds a value of the struct or array `type` from its elements, in order.
Value createAggregate(OpBuilder &builder, Location loc, Type type, ValueRange elements) {
    Value aggregate = LLVM::PoisonOp::create(builder, loc, type);
    for (auto [index, element] : llvm::enumerate(elements))
        aggregate = LLVM::InsertValueOp::create(builder, loc, aggregate, element,
                                                static_cast<int64_t>(index));
    return aggregate;
}

// Lowers `p4hir.struct` and `p4hir.array`, which list all the elements in order.
template <typename Op>
struct AggregateOpConversion : public ConvertOpToLLVMPattern<Op> {
    using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<Op>::OpAdaptor;

    LogicalResult matchAndRewrite(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto type = this->getTypeConverter()->convertType(op.getType());
        if (!type) return rewriter.notifyMatchFailure(op, "unsupported aggregate type");

        rewriter.replaceOp(op, createAggregate(rewriter, op.getLoc(), type, adaptor.getInput()));
        return success();
    }
};

// Returns true if `attr` is zero, false or an aggregate of such constants.
bool isZeroConstant(Attribute attr) {
    return llvm::TypeSwitch<Attribute, bool>(attr)
        .Case<P4HIR::AggAttr>(
            [](auto aggregate) { return llvm::all_of(aggregate.getFields(), isZeroConstant); })
        .Case<P4HIR::IntAttr>([](auto integer) { return integer.isNullValue(); })
        .Case<P4HIR::BoolAttr>([](auto boolean) { return !boolean.getValue(); })
        .Default(false);
}

// Materializes a scalar or aggregate constant.
Value materializeConstant(OpBuilder &builder, Location loc, const TypeConverter &converter,
                          TypedAttr attr) {
    if (auto aggregate = dyn_cast<P4HIR::AggAttr>(attr)) {
        auto type = converter.convertType(attr.getType());
        if (!type) return {};
        if (isZeroConstant(aggregate)) return LLVM::ZeroOp::create(builder, loc, type);

        SmallVector<Value> elements;
        for (auto field : aggregate.getFields().getAsRange<TypedAttr>()) {
            auto value = materializeConstant(builder, loc, converter, field);
            if (!value) return {};
            elements.push_back(value);
        }
        return createAggregate(builder, loc, type, elements);
    }

    if (P4HIR::isZeroWidth(attr.getType()))
        return LLVM::ZeroOp::create(builder, loc, converter.convertType(attr.getType()));

    auto converted = converter.convertTypeAttribute(attr.getType(), attr);
    if (!converted) return {};
    return LLVM::ConstantOp::create(builder, loc, cast<TypedAttr>(*converted));
}

struct ConstOpConversion : public ConvertOpToLLVMPattern<P4HIR::ConstOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto value =
            materializeConstant(rewriter, op.getLoc(), *getTypeConverter(), adaptor.getValue());
        if (!value) return rewriter.notifyMatchFailure(op, "unsupported constant type");
        rewriter.replaceOp(op, value);
        return success();
    }
};

struct StructExtractOpConversion : public ConvertOpToLLVMPattern<P4HIR::StructExtractOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::StructExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!isa<LLVM::LLVMStructType>(adaptor.getInput().getType()))
            return rewriter.notifyMatchFailure(op, "unsupported struct type");

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          op.getFieldIndex());
        return success();
    }
};

// Returns whether the lowered values `lhs` and `rhs` of the P4 type `type` are equal. Structs are
// equal if all their fields are, arrays if all their elements are.
Value createEqual(OpBuilder &builder, Location loc, Type type, Value lhs, Value rhs) {
    if (auto aliasType = dyn_cast<P4HIR::AliasType>(type)) type = aliasType.getCanonicalType();
    if (P4HIR::isZeroWidth(type))
        return LLVM::ConstantOp::create(builder, loc, builder.getI1Type(), 1);

    SmallVector<Type> elementTypes;
    if (auto structType = dyn_cast<P4HIR::StructType>(type)) {
        for (const auto &field : structType.getElements()) elementTypes.push_back(field.type);
    } else if (auto arrayType = dyn_cast<P4HIR::ArrayType>(type)) {
        elementTypes.assign(arrayType.getSize(), arrayType.getElementType());
    } else if (isa<P4HIR::BitsType, P4HIR::BoolType>(type)) {
        return LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::eq, lhs, rhs);
    } else {
        return {};
    }

    Value equal;
    for (auto [index, elementType] : llvm::enumerate(elementTypes)) {
        auto position = static_cast<int64_t>(index);
        Value lhsElement = LLVM::ExtractValueOp::create(builder, loc, lhs, position);
        Value rhsElement = LLVM::ExtractValueOp::create(builder, loc, rhs, position);
        Value elementEqual = createEqual(builder, loc, elementType, lhsElement, rhsElement);
        if (!elementEqual) return {};
        equal = equal ? LLVM::AndOp::create(builder, loc, equal, elementEqual) : elementEqual;
    }
    // Values of empty structs and arrays are all equal.
    if (!equal) equal = LLVM::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    return equal;
}

// Only equality is defined on structs and arrays.
struct AggregateCmpOpConversion : public ConvertOpToLLVMPattern<P4HIR::CmpOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto type = op.getLhs().getType();
        if (auto aliasType = dyn_cast<P4HIR::AliasType>(type)) type = aliasType.getCanonicalType();
        if (!isa<P4HIR::StructType, P4HIR::ArrayType>(type))
            return rewriter.notifyMatchFailure(op, "not a struct or array comparison");
        if (!getTypeConverter()->convertType(type))
            return rewriter.notifyMatchFailure(op, "unsupported cmp operand type");

        auto kind = op.getKind();
        if (kind != P4HIR::CmpOpKind::Eq && kind != P4HIR::CmpOpKind::Ne)
            return rewriter.notifyMatchFailure(op, "only equality is defined on aggregates");

        auto loc = op.getLoc();
        auto equal = createEqual(rewriter, loc, type, adaptor.getLhs(), adaptor.getRhs());
        if (!equal) return rewriter.notifyMatchFailure(op, "unsupported cmp operand type");
        if (kind == P4HIR::CmpOpKind::Ne) {
            Value one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
            equal = LLVM::XOrOp::create(rewriter, loc, equal, one);
        }
        rewriter.replaceOp(op, equal);
        return success();
    }
};

// Returns the value of `index` if it is a constant within the bounds of an array of `size`
// elements.
std::optional<uint64_t> getConstantIndexInBounds(Value index, uint64_t size) {
    llvm::APSInt value;
    if (!matchPattern(index, ::m_ConstantInt(&value))) return std::nullopt;
    if (value.isNegative() || value.uge(size)) return std::nullopt;
    return value.getZExtValue();
}

bool isSignedIndex(Value index) {
    auto bitsType = dyn_cast<P4HIR::BitsType>(index.getType());
    return bitsType && bitsType.isSigned();
}

LLVM::GEPArg getConstantGEPIndex(OpBuilder &builder, Location loc, Type indexType, uint64_t index) {
    // GEP immediates use a signed field. Larger indices use SSA operands.
    if (llvm::isUInt<LLVM::kGEPConstantBitWidth - 1>(index)) return static_cast<int32_t>(index);
    return LLVM::ConstantOp::create(builder, loc, indexType, index).getResult();
}

struct ArrayGetOpConversion : public ConvertOpToLLVMPattern<P4HIR::ArrayGetOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ArrayGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto arrayType = dyn_cast<LLVM::LLVMArrayType>(adaptor.getInput().getType());
        if (!arrayType) return rewriter.notifyMatchFailure(op, "unsupported array type");

        // Dynamic value indexing is unsupported.
        llvm::APSInt value;
        if (!matchPattern(op.getIndex(), ::m_ConstantInt(&value)))
            return rewriter.notifyMatchFailure(op, "expected a constant index");

        auto index = getConstantIndexInBounds(op.getIndex(), arrayType.getNumElements());
        if (!index) {
            // P4 leaves OOB reads unspecified. Choose zero, also for empty arrays.
            rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, arrayType.getElementType());
            return success();
        }
        // LLVM's extractvalue positions are unsigned 32-bit integers.
        if (!llvm::isUInt<32>(*index))
            return rewriter.notifyMatchFailure(op,
                                               "index exceeds LLVM extractvalue position range");

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          static_cast<int64_t>(*index));
        return success();
    }
};

// Returns the LLVM constant for the default value of `type`, or null if it has none. That of a
// struct or array stays a P4HIR aggregate constant, which has no LLVM attribute counterpart.
TypedAttr getDefaultValueAttr(Type type, const TypeConverter &converter) {
    if (auto defaultValueType = dyn_cast<P4HIR::HasDefaultValue>(type)) {
        if (auto defaultValue = defaultValueType.getDefaultValue()) {
            if (isa<P4HIR::AggAttr>(defaultValue))
                return converter.convertType(type) ? defaultValue : TypedAttr();
            return dyn_cast_if_present<TypedAttr>(
                converter.convertTypeAttribute(type, defaultValue).value_or(Attribute()));
        }
    }
    return {};
}

// Materializes a default value returned by `getDefaultValueAttr`.
Value createDefaultValue(OpBuilder &builder, Location loc, const TypeConverter &converter,
                         TypedAttr defaultValue) {
    if (auto aggAttr = dyn_cast<P4HIR::AggAttr>(defaultValue))
        return materializeConstant(builder, loc, converter, aggAttr);
    return LLVM::ConstantOp::create(builder, loc, defaultValue);
}

// P4 leaves the value of an uninitialized variable unspecified. Like mem2reg on P4HIR, the
// lowering takes the default value of its type, rather than leaving the alloca undefined, unless
// `initializeVariables` is off.
struct VariableOpConversion : public ConvertOpToLLVMPattern<P4HIR::VariableOp> {
    VariableOpConversion(const LLVMTypeConverter &converter, bool initializeVariables)
        : ConvertOpToLLVMPattern(converter), initializeVariables(initializeVariables) {}

    LogicalResult matchAndRewrite(P4HIR::VariableOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!isa_and_present<P4HIR::FuncOp>(op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()))
            return rewriter.notifyMatchFailure(op, "not in the body of a function");

        auto objectType = getTypeConverter()->convertType(op.getObjectType());
        if (!objectType) return rewriter.notifyMatchFailure(op, "unsupported object type");

        // A variable with `init` is initialized by its first use. One with a marked lifetime
        // takes its default value where its lifetime starts. The empty struct representing
        // bit<0> holds the only bit<0> value without being stored.
        TypedAttr defaultValue;
        if (initializeVariables && !op.getInit() && !P4HIR::isZeroWidth(op.getObjectType()) &&
            llvm::none_of(op->getUsers(), llvm::IsaPred<P4HIR::LifetimeStartOp>)) {
            defaultValue = getDefaultValueAttr(op.getObjectType(), *getTypeConverter());
            if (!defaultValue) return rewriter.notifyMatchFailure(op, "no default value");
        }

        auto loc = op.getLoc();
        auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(), 1);
        auto alloca = LLVM::AllocaOp::create(rewriter, loc, getPtrType(), objectType, one);
        if (defaultValue) {
            auto value = createDefaultValue(rewriter, loc, *getTypeConverter(), defaultValue);
            LLVM::StoreOp::create(rewriter, loc, value, alloca);
        }
        rewriter.replaceOp(op, alloca);
        return success();
    }

    // Whether to store the default value into variables without an initializer.
    bool initializeVariables;
};

// LLVM only accepts lifetime markers on allocas: those of variables that are not lowered stay.
FailureOr<LLVM::AllocaOp> getLoweredVariable(Operation *op, Value ref,
                                             ConversionPatternRewriter &rewriter) {
    if (auto alloca = ref.getDefiningOp<LLVM::AllocaOp>()) return alloca;
    return rewriter.notifyMatchFailure(op, "variable is not lowered");
}

// A variable comes into existence holding the default value of its type.
struct LifetimeStartOpConversion : public ConvertOpToLLVMPattern<P4HIR::LifetimeStartOp> {
    LifetimeStartOpConversion(const LLVMTypeConverter &converter, bool initializeVariables)
        : ConvertOpToLLVMPattern(converter), initializeVariables(initializeVariables) {}


    LogicalResult matchAndRewrite(P4HIR::LifetimeStartOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto alloca = getLoweredVariable(op, adaptor.getRef(), rewriter);
        if (failed(alloca)) return failure();

        // A variable with `init` is initialized by its first use. The empty struct representing
        // bit<0> holds the only bit<0> value without being stored.
        auto variable = op.getRef().getDefiningOp<P4HIR::VariableOp>();
        auto objectType = cast<P4HIR::ReferenceType>(op.getRef().getType()).getObjectType();
        TypedAttr defaultValue;
        if (initializeVariables && (!variable || !variable.getInit()) &&
            !P4HIR::isZeroWidth(objectType)) {
            defaultValue = getDefaultValueAttr(objectType, *getTypeConverter());
            if (!defaultValue) return rewriter.notifyMatchFailure(op, "no default value");
        }

        auto loc = op.getLoc();
        LLVM::LifetimeStartOp::create(rewriter, loc, *alloca);
        if (defaultValue) {
            auto value = createDefaultValue(rewriter, loc, *getTypeConverter(), defaultValue);
            LLVM::StoreOp::create(rewriter, loc, value, *alloca);
        }
        rewriter.eraseOp(op);
        return success();
    }
    // Whether to store the default value into variables without an initializer.
    bool initializeVariables;
};

struct LifetimeEndOpConversion : public ConvertOpToLLVMPattern<P4HIR::LifetimeEndOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::LifetimeEndOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto alloca = getLoweredVariable(op, adaptor.getRef(), rewriter);
        if (failed(alloca)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::LifetimeEndOp>(op, *alloca);
        return success();
    }
};

// Accesses through packed struct fields use byte alignment, including array elements.
unsigned getAccessAlignment(Value ref) {
    while (auto elementRef = ref.getDefiningOp<P4HIR::ArrayElementRefOp>())
        ref = elementRef.getInput();
    return ref.getDefiningOp<P4HIR::StructFieldRefOp>() ? 1 : 0;
}

// Returns a use of `ref` that a bounds check cannot guard, or null if there is none.
Operation *findUnguardableUse(Value ref) {
    for (Operation *user : ref.getUsers()) {
        // Before it calls a pattern, the conversion casts each operand whose producer stays
        // unconverted. The cast is not a real use: the operation being converted still uses
        // `ref` and is checked itself.
        if (isa<UnrealizedConversionCastOp>(user)) continue;
        if (isa<P4HIR::ReadOp, P4HIR::AssignOp>(user)) {
            if (!isa<P4HIR::FuncOp>(user->getBlock()->getParentOp())) return user;
        } else if (isa<P4HIR::StructFieldRefOp, P4HIR::ArrayElementRefOp>(user)) {
            if (Operation *use = findUnguardableUse(user->getResult(0))) return use;
        } else {
            return user;
        }
    }
    return nullptr;
}

using ArrayBoundsCheck = std::pair<Value, uint64_t>;  // Index and array size.

// Collect the bounds checks needed for a reference chain. Reject the whole
// chain if a potentially out-of-bounds projection has uses we cannot guard.
FailureOr<SmallVector<ArrayBoundsCheck>> collectArrayBoundsChecks(Value ref, Operation *op,
                                                                  RewriterBase &rewriter) {
    SmallVector<ArrayBoundsCheck> checks;

    while (Operation *def = ref.getDefiningOp()) {
        if (auto field = dyn_cast<P4HIR::StructFieldRefOp>(def)) {
            ref = field.getInput();
            continue;
        }
        auto element = dyn_cast<P4HIR::ArrayElementRefOp>(def);
        if (!element) break;

        ref = element.getInput();
        auto index = element.getIndex();
        auto array =
            cast<P4HIR::ArrayType>(cast<P4HIR::ReferenceType>(ref.getType()).getObjectType());
        if (getConstantIndexInBounds(index, array.getSize())) continue;

        if (!isa<P4HIR::BitsType>(index.getType())) {
            return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
                diag << "array index that may be out of bounds has type " << index.getType()
                     << ", which has no LLVM integer type";
            });
        }
        if (!isa<P4HIR::FuncOp>(element->getBlock()->getParentOp())) {
            return rewriter.notifyMatchFailure(
                op,
                "array element reference that may be out of bounds is not directly in a "
                "p4hir.func body");
        }
        if (Operation *use = findUnguardableUse(element.getResult())) {
            return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
                diag << "array element reference that may be out of bounds has a use that "
                        "cannot be guarded: '"
                     << use->getName() << "'";
                if (isa<P4HIR::ReadOp, P4HIR::AssignOp>(use))
                    diag << " not directly in a p4hir.func body";
            });
        }

        checks.push_back({index, array.getSize()});
    }
    return checks;
}

// For each array projection in the reference chain that needs a bounds check:
//   inBounds = icmp ult index, size
//   cond = cond ? (and cond, inBounds) : inBounds
FailureOr<Value> createAccessCondition(Operation *op, Value ref,
                                       ConversionPatternRewriter &rewriter, IntegerType indexType) {
    auto checks = collectArrayBoundsChecks(ref, op, rewriter);
    if (failed(checks)) return failure();

    auto loc = op->getLoc();
    Value condition;
    for (auto [sourceIndex, size] : *checks) {
        Value index = rewriter.getRemappedValue(sourceIndex);
        if (!index) return rewriter.notifyMatchFailure(op, "array index is not converted yet");
        if (P4HIR::isZeroWidth(sourceIndex.getType()))
            index = LLVM::ConstantOp::create(rewriter, loc, indexType, 0);
        auto sourceType = cast<IntegerType>(index.getType());
        if (sourceType.getWidth() < indexType.getWidth())
            index = createIntCast(index, indexType, loc, rewriter, isSignedIndex(sourceIndex));
        Value sizeValue = LLVM::ConstantOp::create(rewriter, loc, index.getType(), size);
        Value inBounds =
            LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ult, index, sizeValue);
        condition = condition ? LLVM::AndOp::create(rewriter, loc, condition, inBounds).getResult()
                              : inBounds;
    }
    return condition;
}

// Lowers a read or a write through a reference.
// `emitAccess` creates the load or the store and returns the loaded value, or null for a store.
// A null `condition` means that the access needs no guard. Otherwise the access only executes
// if `condition` holds, and a read that does not execute yields zero.
//
// Guarded read:
//      zero = llvm.mlir.zero : resultType
//      llvm.cond_br condition, ^load, ^cont(zero)
//   ^load:
//      loaded = llvm.load ptr
//      llvm.br ^cont(loaded)
//  ^cont(result):
//      ...  // result replaces the original read
//
// Guarded write:
//      llvm.cond_br condition, ^store, ^cont
//  ^store:
//      llvm.store ptr, value
//      llvm.br ^cont
//  ^cont:
//
Value createGuardedAccess(ConversionPatternRewriter &rewriter, Operation *op, Value condition,
                          function_ref<Value()> emitAccess) {
    if (!condition) return emitAccess();

    auto loc = op->getLoc();
    Block *head = op->getBlock();
    Block *continuation = rewriter.splitBlock(head, op->getIterator());
    Block *access = rewriter.createBlock(continuation);
    Value loaded = emitAccess();
    SmallVector<Value, 1> accessValues, skipValues;
    Value result;
    if (loaded) {
        accessValues.push_back(loaded);
        result = continuation->addArgument(loaded.getType(), loc);
    }
    LLVM::BrOp::create(rewriter, loc, accessValues, continuation);

    rewriter.setInsertionPointToEnd(head);
    if (loaded) skipValues.push_back(LLVM::ZeroOp::create(rewriter, loc, loaded.getType()));
    LLVM::CondBrOp::create(rewriter, loc, condition, access, ValueRange{}, continuation,
                           skipValues);
    return result;
}

struct ReadOpConversion : public ConvertOpToLLVMPattern<P4HIR::ReadOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ReadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto resultType = getTypeConverter()->convertType(op.getType());
        if (!resultType) return rewriter.notifyMatchFailure(op, "unsupported result type");

        auto condition =
            createAccessCondition(op, op.getRef(), rewriter, cast<IntegerType>(getIndexType()));
        if (failed(condition)) return failure();

        Value result = createGuardedAccess(rewriter, op, *condition, [&]() -> Value {
            return LLVM::LoadOp::create(rewriter, op.getLoc(), resultType, adaptor.getRef(),
                                        getAccessAlignment(op.getRef()));
        });
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct AssignOpConversion : public ConvertOpToLLVMPattern<P4HIR::AssignOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto condition =
            createAccessCondition(op, op.getRef(), rewriter, cast<IntegerType>(getIndexType()));
        if (failed(condition)) return failure();

        createGuardedAccess(rewriter, op, *condition, [&]() -> Value {
            LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), adaptor.getRef(),
                                  getAccessAlignment(op.getRef()));
            return {};
        });
        rewriter.eraseOp(op);
        return success();
    }
};

struct StructFieldRefOpConversion : public ConvertOpToLLVMPattern<P4HIR::StructFieldRefOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::StructFieldRefOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (failed(collectArrayBoundsChecks(op.getInput(), op, rewriter))) return failure();

        auto objectType = cast<P4HIR::ReferenceType>(op.getInput().getType()).getObjectType();
        auto structType = getTypeConverter()->convertType(objectType);
        if (!structType) return rewriter.notifyMatchFailure(op, "unsupported struct type");

        auto fieldIndex = static_cast<int32_t>(op.getFieldIndex());
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, getPtrType(), structType, adaptor.getInput(),
                                                 ArrayRef<LLVM::GEPArg>{0, fieldIndex},
                                                 LLVM::GEPNoWrapFlags::inbounds);
        return success();
    }
};

struct ArrayElementRefOpConversion : public ConvertOpToLLVMPattern<P4HIR::ArrayElementRefOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(P4HIR::ArrayElementRefOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (failed(collectArrayBoundsChecks(op.getResult(), op, rewriter))) return failure();

        auto objectType = cast<P4HIR::ReferenceType>(op.getInput().getType()).getObjectType();
        auto arrayType =
            dyn_cast_if_present<LLVM::LLVMArrayType>(getTypeConverter()->convertType(objectType));
        if (!arrayType) return rewriter.notifyMatchFailure(op, "unsupported array type");

        auto createGEP = [&](LLVM::GEPArg index) {
            return rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
                op, getPtrType(), arrayType, adaptor.getInput(), ArrayRef<LLVM::GEPArg>{0, index},
                LLVM::GEPNoWrapFlags::inbounds);
        };

        if (auto index = getConstantIndexInBounds(op.getIndex(), arrayType.getNumElements())) {
            createGEP(getConstantGEPIndex(rewriter, op.getLoc(), getIndexType(), *index));
            return success();
        }

        auto loc = op.getLoc();
        Value indexValue = adaptor.getIndex();
        if (P4HIR::isZeroWidth(op.getIndex().getType()))
            indexValue = LLVM::ConstantOp::create(rewriter, loc, getIndexType(), 0);
        if (!isa<IntegerType>(indexValue.getType()))
            return rewriter.notifyMatchFailure(op, "unsupported index type");

        auto indexType = cast<IntegerType>(getIndexType());
        Value index =
            createIntCast(indexValue, indexType, loc, rewriter, isSignedIndex(op.getIndex()));

        // An out-of-bounds GEP may produce poison. The reference can only feed
        // further projections and accesses guarded by the original indices.
        createGEP(index);
        return success();
    }
};

struct LowerP4HIRToLLVMPass : public P4::P4MLIR::impl::LowerP4HIRToLLVMBase<LowerP4HIRToLLVMPass> {
    using LowerP4HIRToLLVMBase::LowerP4HIRToLLVMBase;
    void runOnOperation() override {
        auto &context = getContext();
        auto module = getOperation();

        LLVMTypeConverter typeConverter(&context);
        populateP4HIRToLLVMTypeConversion(typeConverter);

        LLVMConversionTarget target(context);
        target.addLegalOp<ModuleOp>();

        RewritePatternSet patterns(&context);
        populateP4HIRToLLVMConversionPatterns(typeConverter, patterns, initializeVariables);

        // Lowering should be driven by the patterns above, not by constant
        // folding P4HIR ops (e.g. `p4hir.binop(add, ...)` on two constants)
        // before they even get a chance to match.
        ConversionConfig config;
        config.foldingMode = DialectConversionFoldingMode::Never;

        if (failed(applyPartialConversion(module, target, std::move(patterns), config))) {
            signalPassFailure();
        }
    }
};

}  // namespace

void P4::P4MLIR::populateP4HIRToLLVMTypeConversion(LLVMTypeConverter &converter) {
    converter.addConversion([](P4HIR::BitsType bitsType) -> Type {
        // LLVM has no `i0`. Like `bit<0>`, the empty struct `{}` has no bits and a single value.
        // Operations on `bit<0>` are expected to be folded before lowering, leaving only
        // constants, storage and control flow of this type.
        if (bitsType.getWidth() == 0)
            return LLVM::LLVMStructType::getLiteral(bitsType.getContext(), {});
        return IntegerType::get(bitsType.getContext(), bitsType.getWidth());
    });

    converter.addConversion(
        [](P4HIR::BoolType boolType) { return IntegerType::get(boolType.getContext(), 1); });

    // References lower to opaque pointers, as the memory operations carry the object type.
    // A reference to an object without an LLVM counterpart has none either.
    converter.addConversion([&converter](P4HIR::ReferenceType refType) -> std::optional<Type> {
        if (!converter.convertType(refType.getObjectType())) return std::nullopt;
        return LLVM::LLVMPointerType::get(refType.getContext());
    });

    // Structs lower to literal packed LLVM structs with the fields in the same order, so that
    // field indices carry over, and without padding between fields.
    converter.addConversion([&converter](P4HIR::StructType structType) -> std::optional<Type> {
        SmallVector<Type> fieldTypes;
        for (const auto &field : structType.getElements()) {
            auto fieldType = converter.convertType(field.type);
            if (!fieldType) return std::nullopt;
            fieldTypes.push_back(fieldType);
        }
        return LLVM::LLVMStructType::getLiteral(structType.getContext(), fieldTypes,
                                                /*isPacked=*/true);
    });

    converter.addConversion([&converter](P4HIR::ArrayType arrayType) -> std::optional<Type> {
        auto elementType = converter.convertType(arrayType.getElementType());
        if (!elementType) return std::nullopt;
        return LLVM::LLVMArrayType::get(elementType, arrayType.getSize());
    });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BitsType bitsType,
                     P4HIR::IntAttr attr) -> LLVMTypeConverter::AttributeConversionResult {
            // bit<0> lowers to an empty struct, so it cannot use an IntegerAttr.
            if (auto convertedType = dyn_cast<IntegerType>(converter.convertType(bitsType))) {
                return IntegerAttr::get(convertedType, attr.getValue());
            }
            return LLVMTypeConverter::AttributeConversionResult::na();
        });

    converter.addTypeAttributeConversion(
        [&converter](P4HIR::BoolType boolType, P4HIR::BoolAttr attr) {
            return IntegerAttr::get(converter.convertType(boolType), attr.getValue() ? 1 : 0);
        });
}

void P4::P4MLIR::populateP4HIRToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                       RewritePatternSet &patterns,
                                                       bool initializeVariables) {
    patterns.add<ConstOpConversion, BinOpConversion, UnaryOpConversion, CmpOpConversion,
                 BrOpConversion, CondBrOpConversion, ConcatOpConversion, SliceOpConversion,
                 CastOpConversion, ReadOpConversion, AssignOpConversion, LifetimeEndOpConversion>(
        converter);
    patterns.add<AggregateOpConversion<P4HIR::StructOp>, AggregateOpConversion<P4HIR::ArrayOp>,
                 AggregateCmpOpConversion, StructExtractOpConversion, ArrayGetOpConversion>(
        converter);
    patterns.add<VariableOpConversion, LifetimeStartOpConversion>(converter, initializeVariables);
    patterns.add<StructFieldRefOpConversion, ArrayElementRefOpConversion>(converter);
}
