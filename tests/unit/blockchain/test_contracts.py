import pytest
from brownie import accounts, CarbonCredits
from web3 import Web3

@pytest.fixture(scope="module")
def contract():
    return accounts[0].deploy(CarbonCredits)

def test_contract_deployment(contract):
    assert contract.name() == "EcoTrackCarbon"
    assert contract.symbol() == "ECC"
    assert contract.totalSupply() == 0

def test_mint_permissions(contract):
    with pytest.reverts("AccessControl: account is missing role"):
        contract.mint(accounts[1], 100, {"from": accounts[1]})

def test_credit_transfer(contract):
    contract.mint(accounts[0], 1000, {"from": accounts[0]})
    contract.transfer(accounts[1], 500, {"from": accounts[0]})
    
    assert contract.balanceOf(accounts[0]) == 500
    assert contract.balanceOf(accounts[1]) == 500

def test_event_emission(contract):
    tx = contract.mint(accounts[0], 100, {"from": accounts[0]})
    assert "CreditsMinted" in tx.events
    assert tx.events["CreditsMinted"]["to"] == accounts[0].address
