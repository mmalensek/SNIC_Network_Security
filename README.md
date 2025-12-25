# Smart Network Interface Cards for in-network AI Network Security
In this explorative diploma thesis I am investigating the capabilities of **Smart NIC's**, their integration with the hosting server, and the SDKs used for controlling the cards.
I am also implementing a case study of in-network computation harnessing the GPU of the **A30X**, in this case the computation involves machine learning-based **network traffic filtering**.

## Running of the code

Make sure, to have installed a valid dataset and ollama on your machine.

To run the code, first make sure to make "run_test.sh" an executable (with **"chmod +x run_test.sh"**) and secondly simply run **"./run_test.sh"**.

To note, the automated testing can take a very long time, depending on the installed models.
So I recommend also using **tmux**, especially for SSH connections. 

How to use tmux? 
1) **"tmux new -s testingt"** - to start a tmux session
2) **"./run_test.sh"** - run bash normally
3) **"tmux attach -t testing"** - reattach to tmux session, incase of disconnecting 

## Keywords
Computer networks, Software-defined networking, GPUs, Machine learning, Network Security.

## Acknowledgements 
Mentor izr. prof. dr. Veljko Pejović and asist. Miha Grohar

## Author
[@mmalensek](https://github.com/mmalensek)
