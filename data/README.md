# Gathering Data

Previous explorations of the data have shown that historical trends in LDZ gas demand do not change significantly. We have therefor chose to limit historical data to the period from January 2019 up to November 2022. 

If you managed to create your python environment as per the instructions in the main _README.md_ then you are ready to now also collect the data. Simply run `python src/get_data.py` and it will gather the necessary datasets and place them in the **data** folder.

You can adjust the time periods for which to download data by changing the dates in `get_data.py` (line 132-133).

## CWV

Gas demand in the UK varies depending on a number of factors. One main factor is the weather and in particular temperature and wind speed. The Composite Weather Variable (CWV) is defined so that weather is accounted for when estimating gas demand. We pull this data from the [Data Item Explorer / MIPI](https://mip-prd-web.azurewebsites.net/DataItemExplorer/Index). For more information about the CWV see this [document](https://www.gasgovernance.co.uk/sites/default/files/ggf/2019-03/CWV%20-%20Temp%20and%20Wind%20Speed%20Analysis.pdf) by Xoserve. 

## MIPI Gas Actuals

Historical or actual gas demand is published on the MIPI platform on a daily basis. It is split up in a number of components and we download all of them even though we are only interested in the (Local Distribution Zone) LDZ component. A LDZ represents gas demand from individual consumers and the UK is split up into 13 LDZs. 