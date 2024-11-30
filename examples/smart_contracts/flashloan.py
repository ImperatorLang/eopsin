from opshin.prelude import *
from opshin.std.fractions import *


@dataclass
class FlashloanParams(PlutusData):
    CONSTR_ID = 0
    owner: PubKeyHash
    loan_token: Token
    loan_fee: Fraction


@dataclass
class FlashloanState(PlutusData):
    CONSTR_ID = 0
    params: FlashloanParams
    spent_for: TxOutRef


@dataclass
class FlashLend(PlutusData):
    CONSTR_ID = 0
    input_index: int
    output_index: int


@dataclass
class FlashWithdraw(PlutusData):
    CONSTR_ID = 1


def validator(
    state: FlashloanState,
    redeemer: Union[FlashLend, FlashWithdraw],
    context: ScriptContext,
) -> None:
    tx_info = context.tx_info
    purpose: Spending = context.purpose
    if isinstance(redeemer, FlashWithdraw):
        assert state.params.owner in tx_info.signatories, "Only the owner can withdraw"
    elif isinstance(redeemer, FlashLend):
        own_input_info = tx_info.inputs[redeemer.input_index]
        assert (
            own_input_info.out_ref == purpose.tx_out_ref
        ), "Input index must match the spent utxo"
        own_input = own_input_info.resolved
        own_output = tx_info.outputs[redeemer.output_index]
        assert (
            own_output.address == own_input.address
        ), "Output must be sent back to contract"
        assert own_output.datum == SomeOutputDatum(
            FlashloanState(state.params, purpose.tx_out_ref)
        )
        loan_token = state.params.loan_token
        for pid, name_amount_dict in own_input.value.items():
            for name, amount in name_amount_dict.items():
                if Token(pid, name) == loan_token:
                    assert ge_fraction(
                        mul_fraction(Fraction(amount, 1), state.params.loan_fee),
                        Fraction(own_output.value[pid][name], 1),
                    ), "Output must satisfy loan fee"
                else:
                    assert (
                        own_output.value[pid][name] == amount
                    ), "Output must have same value as input"
        assert len(own_output.to_cbor()) <= 1000, "Output must be less than 1000 bytes"
    else:
        assert False, "Invalid redeemer"
