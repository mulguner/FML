# FIN-407

Repository for final project in the EPFL course FIN-407. The project report is contained in project report.pdf.

Data can be downloaded separately from
https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts?resource=download.

Unzip the data and paste motley-fool-data.pkl into the FIN-407/data folder in your local environment.

To reproduce our sentiment scores found in data/no_transcript_final_dataset.csv:
1) Follow the instructions in scripts/clean_data.py to create both BoW-tokenized-transcripts.csv and cleaned-dataset.csv.
2) Run data/fix_dataset.ipynb to create final_dataset.csv, which removes transcripts belonging to firms with no WRDS earnings.
3) Assign scores to the transcripts in final_dataset by: 
    * First, run scripts/bigram-tf-id.ipynb to add tf-idf scores to each transcript.
    * Then, run scripts/bow.ipynb to assign BoW scores to each transcript, simultaneously creating data/no_transcript_final_dataset.csv.
    * Finally, run vader.ipynb to add VADER scores to data/final_dataset.csv and data/no_transcript_final_dataset.csv.
4) Feel free to run scripts/finbert.ipynb to add FinBERT scores to your transcripts, which we did not do due to computing power constraints.

For exploratory data analysis, see eda.ipynb, which can only be run after the above.

(Alternatively, download final_dataset.csv, cleaned-dataset.csv and BoW-tokenized-transcripts.csv from Dropbox here: https://www.dropbox.com/scl/fi/nj2qh8rxspkg9ucwf1aqd/data.zip?rlkey=8b4193eyodddmbexe6iq3y5b8&st=om2q1w6f&dl=0 and place them in the data folder.)

To reproduce the decile portfolios and trading strategies: 
1) Run make_portfolio.ipynb to create the portfolios in data/portfolio_decile.
2) Run portfolio_returns.ipynb to create time series of the decile portfolio returns, found in data/portfolio_decile_value
2) Investigate decile portfolio performance using performance_decile.ipynb, with visualizations saved in /portfolio_decile_vis.
3) Investigate long-short strategy performance using performance.ipynb, also saved in data/portfolio_decile_vis.

Finally, run lda.ipynb to reproduce our LDA results.

Ulrik T. Sjøli, Mert Ülgüner and Zhuofu Zhou, Lausanne, 2024
