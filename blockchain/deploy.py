#!/usr/bin/env python3
import os
from brownie import accounts, network, config, CarbonCredits
from pathlib import Path

def main():
    # Get network configuration
    network_name = network.show_active()
    print(f"Deploying to {network_name} network")
    
    # Load account
    if network_name in ["development", "ganache-local"]:
        deployer = accounts[0]
    else:
        deployer = accounts.add(config["wallets"]["from_key"])
    
    # Deploy contract
    contract = CarbonCredits.deploy(
        {'from': deployer},
        publish_source=config["networks"][network_name].get("verify", False)
    )
    
    print(f"Contract deployed at {contract.address}")
    
    # Save deployment info
    with open(Path("deployments") / f"{network_name}.txt", "w") as f:
        f.write(contract.address)
    
    return contract.address

if __name__ == "__main__":
    main()
