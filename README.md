# API
The Little Andaman folder can be divided into to parts,
  1. The files outside of Little Andaman/API folder are the original files of the project, with the
       * depri.ipynb = main jupyter notebook where the model was initially fine tuned using rag (updated with chat history, tho integration isn't fully complete)
       * D_set & V_str with the dataset and vector store made out of it
       * cache files/folders
       * Few test .py files for code and api

  2. The ##API folder constitues the most recent progress so far
       * It has a copy of the same D_set & V_set folders and cache folder created from the code
       * The main code from jupyter notebook broken into three individual python files which import the functions innetween them.
       * Two more api teststing python files are in the folder
