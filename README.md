# AI Assistant Luminia

This program is made to run on windows or linux, depending on what platform it runs on it requires some libraries that are only compatible with that specific OS, all the libraries that are needed to run the program are in the two .txt files named pip install, there's a conflict in the packages themselves when trying to install them all at once, that's why there's no requirements.txt file you can use, you have to manually install them one by one, that's how it works for some reason

This project is meant to be used in a Raspberry Pi to send signals on its pin to make servos move around essentially giving hands to the AI, a button is also used in to read when the user is intended to talk to it and begin all the process the program does, for more details on how this robot came to be built you can check out the Office or PDF document for further in depth details of all considerations, steps and decision that where made to make this project (the documents are in Spanish, if you need it in another language you will have to translate it yourself or ask a Spanish speaking person if you know one to translate it for you, sorry, I don't provide a translated version)

In order to try to run this program at all you gotta have a Hugging Face account which you can create from its official website (or edit the code to use your own AI models) and generate a token of access to run inference on the models placed on the code, you can check the code to know which models are being selected, there's a chance these models are taken down, removed or not available for inference online, and therefor you must need to select new ones or download and run them yourself on your computer (WARNING: If you do decide to run them in your computer, you will need to have good performance as this models are heavy and may take long to generate a response or lag your computer a lot)

This program already comes with a virtual enviroment setup with all the packages installed, but it only works on Linux since it was meant to run on the Raspberry Pi, there's no script to run it on windows.

# Steps to run this program:

1. Download and install python 3.11.2: https://www.python.org/downloads/release/python-3112
2. From the terminal run the pip commands to install the packages needed one by one that are located in the "pip command" .txt files depending on your operating system, if you can use the already made enviroment located in the "env" folder (only works on Linux) you don't have to install anything then, skip this step, all you need is use the python file inside the bin folder of the enviroment
3. Create an account on Hugging Face: https://huggingface.co/models and generate an access token for the AI models from your account which has the form of "hf_xxxxx..."
4. Replace the token inside main.py with your token
5. Run main.py

PD: The code has a hugging face token, I forgot to delete it so it has been deactivated, the token in the code doesn't work anymore then
