# Trust-region method for representation learning: MICo integration

This repository implements the trust-region optimization methdod into the MICo repository by Google Research. As new versions of Dopamine and other libraries have been released, the environment to run the code is not given here. Requirements.txt from the original code base is included but doesn't contain all that is required nor the required versions.

Some patches had to be applied, inlcuding but not limited to the gin-files and some utils.

The codebase is based on MICo: Improved representations via sampling-based
state similarity for Markov decision processes by Castro et al. (2022) (https://arxiv.org/pdf/2106.08229).

Their README below:

---

# MICo: Improved representations via sampling-based state similarity for Markov decision processes

Modify `atari/run.sh` to execute code.
