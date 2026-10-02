to clone repo :
git clone https://github.com/YOUR_USERNAME/SentinelXAI.git
main directory: cd SentinelXAI

for large file downloads:
git lfs install
git lfs pull (may take a while)

VIRTUAL ENV very important to activate:

python -m venv .venv
.venv\Scripts\activate

Install requirements:

python -m pip install --upgrade pip
pip install -r requirements.txt

finally, run the dashboard. 

python -m streamlit run app.py




TIPS FOR DEMO:

if we wanna show the successful attack then 

for FGSM choose sample 31
For PGD choose 469 and for 
Deepfool just go through the attacks and u can find successful adversarial attacks.
