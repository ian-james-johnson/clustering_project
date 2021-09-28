Zillow Clustering Project

Project Summary:
Analyze Zillow dataset to find drivers of log error for single family residential properties with transaction dates in 2017.

Project Goals:
Notebook and presentation on drivers of log error.
Use clustering methods for feature engineering.

Reproduction of Project:
The initial_project notebook (contains the work), wrangle.py (contains functions used in the notebook), and env.py files (contains credentials for accessing SQL server) are needed. The notebook can be run in oreder of the fields going downwards.

Data Science Pipeline
Acquire:
Obtained Zillow dataframe from Codeup SQL server.
Save a local csv copy of Zillow dataframe.

Prepare:
Univariate exploration.
Removed rows and columns with too many nulls.
Feature engineering.
Split the data into train, validate, and test.

Explore:
Multivariate explorations.
Statistically tested hypothesises.
Scaled the data.
Performed clustering to identify new features.

Model:
Created a mean baseline model.
Created OLS, LassoLars, and Polynomial models.
Polynomials was selected for use on the test dataset.

Deliver:
Created project notebook for reproducibility and presentation.

Data Dictionary:
acres: sqft area converted into acres (43560 sqft = 1 acre)
age: age of the property
bathrooms: number of bathrooms
bedrooms: number of bedrooms
fips: federal information processing standard code. Identifies the county of the property.
land_dollar_per_sqft: land tax value divided by lot size
land_tax_value: 2017 assessed tax value
latitude: the latitude of the property
logerror: log(Zestimate) - log(SalePrice)
lot_size: area in sqft of property
parcel_id: unique identifier for lots
structure_dollar_per_sqft: structure tax value divided by sqft
structure_tax_value: 2017 assessed tax on structure
square_feet: living area of home
tax_amount: 2017 tax due
taxrate: tax amount divided home value
taxes: taxes assessed for 2017
zip_code: the zip code location of the property
