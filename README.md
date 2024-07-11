# IceTop ML Cosmic Ray Reconstructions
Welcome to icetop-cnn!

## Table of Contents

- [Introduction](#introduction)
- [Installation](#how-to-install)
- [Usage](#user-guide)
- [Known Issues](#known-issues-wip)

## Introduction

## How to Install
### Log in to Cobalt
If you have configured your .ssh, then you may log in with the following command:
``` bash
ssh cobalt
```
Otherwise, you may log in with the following commands:
``` bash
ssh jdoe@pub.icecube.wisc.edu
ssh cobalt
```
Be sure to replace the example username with your actual IceCube username.\
There is a section on [configuring your SSH](#configuring-ssh) further on in this document.
### Clone the GitHub Repository
The next step is to clone the GitHub repository.
``` bash
git clone https://github.com/fmcnallyi3/icetop-cnn.git
```
<div style="text-align: center;">
  :warning:`WARNING`:warning:
</div>
It is discouraged for new users to specify a different name for the directory.\
This is because some file paths depend on this naming convention.\
For more experienced users, edit at your own risk.

## User Guide
### Configuring SSH
New users are recommended to take the time to first configure their .ssh.\
It is also highly recommended to set up an authentication key. This avoids the need to enter your password each time you log in.\
If you are unfamiliar with how to do that, [here](https://www.ssh.com/academy/ssh/keygen) is a link to help get you started.

<div style="text-align: center;">
  :warning:`WARNING`:warning:
</div>
Cobalt has 8 nodes to choose from, 1-8. It is VERY important that you choose one of the latter four nodes, 5-8. \
This is because the former nodes use an older CPU architecture that is incompatable with our machine learning library, [TensorFlow](https://www.tensorflow.org/versions/r2.14/api_docs).\
Below is an example configuration file, complete with entries for an authentication key.
```
# ~/.ssh/config:

# ICECUBE PUBLIC
Host pub
  HostName pub.icecube.wisc.edu
  User jdoe
  IdentityFile ~/.ssh/id_rsa

# COBALT
Host cobalt
  HostName cobalt06.icecube.wisc.edu
  User jdoe
  IdentityFile ~/.ssh/id_rsa
  ProxyJump jdoe@pub.icecube.wisc.edu

# SUBMITTER
Host submit
  HostName submitter.icecube.wisc.edu
  User jdoe
  IdentityFile ~/.ssh/id_rsa
  ProxyJump jdoe@pub.icecube.wisc.edu
```

## Known Issues (WIP)