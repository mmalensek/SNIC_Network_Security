# Explainable network intrusion detection with large language models
In this explorative diploma thesis, I investigate the use of large language models for explainable network intrusion detection. The work focuses on integrating machine learning-based detection methods with LLMs to provide interpretable insights into network traffic anomalies and malicious behavior. As part of the study, I implement a case study that combines traditional intrusion detection techniques with LLM-driven explanations, enabling more transparent and human-understandable network traffic analysis and filtering.

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
