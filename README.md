## BankTermDepositPrediction
Bank Institution Term Deposit Predictive Model

### Business Need
You successfully finished up to your rigorous job interview process with Bank of Portugal as a machine learning researcher. The investment and portfolio department would want to be able to identify their customers who potentially would subscribe to their term deposits. As there has been heightened interest of marketing managers to carefully tune their directed campaigns to the rigorous selection of contacts, the goal of your employer is to find a model that can predict which future clients who would subscribe to their term deposit. Having such an effective predictive model can help increase their campaign efficiency as they would be able to identify customers who would subscribe to their term deposit and thereby direct their marketing efforts to them. This would help them better manage their resources (e.g human effort, phone calls, time)
The Bank of Portugal, therefore, collected a huge amount of data that includes customers profiles of those who have to subscribe to term deposits and the ones who did not subscribe to a term deposit. As their newly employed machine learning researcher, they want you to come up with a robust predictive model that would help them identify customers who would or would not subscribe to their term deposit in the future.
Your main goal as a machine learning researcher is to carry out data exploration, data cleaning, feature extraction, and developing robust machine learning algorithms that would aid them in the department.

### Data and Features
The dataset should be downloaded from the [UCI ML](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  website and more details about the data can be read from the same website. From the website, you would find access to four datasets:
Bank-additional-full CSV is our guy.


### Running the code
* For the notebook, all you need is to upload the csv file to your working environment and execute the cells.
* For the scripts: 
  * clone the repo.
  * Pip install the requirements file.
  * cd into the scripts directory.
  * On the terminal, run the following command to execute and get the predictions into a csv file:
  ```python main.py 'file path'```
  
 **PS**: You can change the default(MultiLayer Perceptrron) model selected in the main.py.
 
 ### Project Structure
`notebooks`contains the following notebooks:

* eda.ipynb -eda notebook (using plotly)
* BankOfPortugal1.ipynb - ML/prediction notebook

`licence`License template

`readme`Markdown file giving a brief description of the project and its structure.

`scripts`contains the following python scripts:

* data.py - preprocessing
* main.py - automation
* model.py - model class
* util.py - ____

`story1.pdf`Interim slides

`requirements.txt`Dependenncies

`dashboard plots`Tableau Dashboard plot shots.

