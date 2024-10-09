BDN Investment Decision Making

This project explores Bayesian Decision Networks (BDNs) for investment decision-making on the Johannesburg Stock Exchange using the INVEST system. It implements different learning algorithms to optimize Conditional Probability Tables (CPTs) for better decision outcomes.

Getting Started

Setup
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/BDN-Investment-Decision.git
cd BDN-Investment-Decision
Create a virtual environment and activate it:
bash
Copy code
python -m venv env
source env/bin/activate  # On Windows: `env\Scripts\activate`
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Running Experiments
You can test different CPT learning algorithms using the following commands:

Maximum Likelihood Estimation (MLE):
bash
Copy code
python app.py --cpt_algorithm MLE
Expectation-Maximization (EM):
bash
Copy code
python app.py --cpt_algorithm EM
Minimum Description Length (MDL):
bash
Copy code
python app.py --cpt_algorithm MDL
Optional Parameters
--start <year>: Start year for evaluation (e.g., --start 2015)
--end <year>: End year for evaluation (e.g., --end 2018)
--noise True: Add noise to the dataset (e.g., --noise True)
Example with options:

bash
Copy code
python app.py --cpt_algorithm MLE --start 2015 --end 2018 --noise True
Output
Results: The results will be printed to the console and saved in the results/ directory.
Logs: Execution logs are saved in the logs/ folder for analysis.
