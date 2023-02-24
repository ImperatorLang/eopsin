from eopsin.prelude import *

# parameters controlling which token is to be wrapped (currently preprod MILK) and how many decimal places to add
TOKEN_POLICYID = b"\xae\x81\x071\xb5\xd2\x1c\r\x18-\x89\xc6\n\x1e\xffp\x95\xdf\xfd\x1c\r\xce\x87\x07\xa8a\x10\x99"
TOKEN_NAME = b"MILK"
WRAPPING_FACTOR = 1000000


@dataclass()
class Empty(PlutusData):
    pass


TOKEN = Token(TOKEN_POLICYID, TOKEN_NAME)


def all_tokens_unlocked_from_contract_address(
    txins: List[TxInInfo], address: Address, token: Token
) -> int:
    # generally always iterate over all inputs to avoid double spending
    res = 0
    for txi in txins:
        if txi.resolved.address == address:
            res += txi.resolved.value.get(token.policy_id, {b"": 0}).get(
                token.token_name, 0
            )
    return res


def own_spent_utxo(txins: List[TxInInfo], p: Spending) -> TxOut:
    # obtain the resolved txout that is going to be spent from this contract address
    for txi in txins:
        if txi.out_ref == p.tx_out_ref:
            own_txout = txi.resolved
    # This throws a name error if the txout was not found
    return own_txout


def own_policy_id(own_spent_utxo: TxOut) -> PolicyId:
    # obtain the policy id for which this contract can validate minting/burning
    cred = own_spent_utxo.address.credential
    if isinstance(cred, ScriptCredential):
        policy_id = cred.validator_hash
    # This throws a name error if the credential is not a ScriptCredential instance
    return policy_id


def own_address(own_policy_id: PolicyId) -> Address:
    return Address(ScriptCredential(own_policy_id), NoStakingCredential())


def all_tokens_locked_at_contract_address(
    txouts: List[TxOut], address: Address, token: Token
) -> int:
    res = 0
    for txo in txouts:
        if txo.address == address:
            res += txo.value.get(token.policy_id, {b"": 0}).get(token.token_name, 0)
            # enforce a small inlined datum
            assert txo.datum == SomeOutputDatum(
                b""
            ), "Does not attach correct datum to script output"
    return res


def validator(_datum: None, _redeemer: NoRedeemer, ctx: ScriptContext) -> None:
    purpose = ctx.purpose
    if isinstance(purpose, Minting):
        # whenever tokens should be burned/minted, the minting purpose will be triggered
        own_addr = own_address(purpose.policy_id)
        own_pid = purpose.policy_id
    elif isinstance(purpose, Spending):
        # whenever something is unlocked from the contract, the spending purpose will be triggered
        own_utxo = own_spent_utxo(ctx.tx_info.inputs, purpose)
        own_pid = own_policy_id(own_utxo)
        own_addr = own_utxo.address
    else:
        assert False, "Incorrect purpose given"
    all_locked = all_tokens_locked_at_contract_address(
        ctx.tx_info.outputs, own_addr, TOKEN
    )
    all_unlocked = all_tokens_unlocked_from_contract_address(
        ctx.tx_info.inputs, own_addr, TOKEN
    )
    all_minted = ctx.tx_info.mint.get(own_pid, {b"": 0}).get(b"w" + TOKEN_NAME, 0)
    print("unlocked from contract: " + str(all_unlocked))
    print("locked at contract: " + str(all_locked))
    print("minted: " + str(all_minted))
    assert (
        (all_locked - all_unlocked) * WRAPPING_FACTOR
    ) == all_minted, "Wrong amount of tokens minted, difference: " + str(
        (all_locked - all_unlocked) * WRAPPING_FACTOR - all_minted
    )
