# Data wrangling
import numpy as np
import pandas as pd

# Visualizing
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Statistical analysis
import scipy.stats as stats

# Modeling
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.cluster import KMeans, dbscan 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score

# Custom user files
# Credentials for logging into Codeup SQL server
from env import user, password, host
# User created functions for wrangling data
#import wrangle

import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Setting up display options for pandas
pd.set_option('display.max_columns', 80)
pd.set_option("precision", 3)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def get_connection(db, user=user, host=host, password=password):
    '''
    This function creates a connection to the Codeup db.
    It takes db argument as a string name.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def new_zillow_data():
    '''
    This function gets new zillow data from the Codeup database.
    '''
    sql_query = """
                SELECT prop.*,
                pred.logerror,
                pred.transactiondate,
                air.airconditioningdesc,
                arch.architecturalstyledesc,
                build.buildingclassdesc,
                heat.heatingorsystemdesc,
                landuse.propertylandusedesc,
                story.storydesc,
                construct.typeconstructiondesc

                FROM   properties_2017 prop
                
                INNER JOIN (SELECT parcelid, Max(transactiondate) transactiondate
                FROM   predictions_2017 GROUP  BY parcelid) pred
                USING (parcelid)
               	
                JOIN predictions_2017 as pred USING (parcelid, transactiondate)
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
                LEFT JOIN storytype story USING (storytypeid)
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                
                WHERE  prop.latitude IS NOT NULL AND prop.longitude IS NOT NULL
                
                Limit 100000
                """
    # Read in dataframe from Codeup SQL database
    df = pd.read_sql(sql_query,get_connection('zillow'))
    return df



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def get_zillow_data():
        '''
        This function gets zillow data from local csv, or otherwise from Codeup SQL database.
        '''
        if os.path.isfile('zillow.csv'):
            df = pd.read_csv('zillow.csv', index_col = 0)
        else:
            df = new_zillow_data()
            df.to_csv('zillow.csv')
        return df



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def nulls_by_col(df):
    '''
    This function takes in a dataframe and returns a dataframe where each row is
    a feature, the first column is the number of rows with missing values,
    and the second column is the percent missing values for that feature.
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows':  num_missing, 'percent_rows_missing': pct_missing})
    return cols_missing



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def handle_missing_values(df, prop_required_column = 0.5, prop_required_row = 0.75):
    '''
    This function takes in a dataframe, the proportion of columns (0-1) with non-missing values,
    and the proportion of rows (0-1) with non_missing values to keep each column or row.
    It returns the dataframe with the specified columns and row dropped.
    '''
    threshold = int(round(prop_required_column * len(df.index), 0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def wrangle_zillow():
    '''
    This function acquires the zillow data and then prepares it.
    Rows and columns with too many missing values are dropped.
    THe dataframe is restricted to only residential properties with
    at least one bedroom and bathroom, and at least 500 sqft.
    Unnecessarry columns are dropped.
    Outliers in taxvaluedollarcnt and calculatedfinishedsquarefeet
    are adjusted for.
    Missing values are filled in with median for buildinglotsize, and buildingquality.
    Columns are renamed for ease of use.
    '''
    df = get_zillow_data()

    # Change fips to int
    df.fips = df.fips.astype(int)

    # Restrict df to only single use properties
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]

    # Restrict df to only properties with at least one bedroom/bathroom and at least 500 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.calculatedfinishedsquarefeet >= 500) & ((df.unitcnt <= 1) | (df.unitcnt.isnull()))]

    # Drop row and columns that have too many missing values
    df = handle_missing_values(df)

    # Add a new column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles', np.where(df.fips == 6059, 'Orange', 'Ventura'))

    # Drop unneeded columns
    df = df.drop(columns=['id', 'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc', 'censustractandblock', 'rawcensustractandblock', 'propertylandusedesc'])

    # Replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace=True)

    # Replace nulls in heating system with none (this is southern California)
    df.heatingorsystemdesc.fillna('None', inplace=True)

    # Replace nulls with median value
    df.buildingqualitytypeid.fillna(6.0, inplace=True)
    df.lotsizesquarefeet.fillna(7313, inplace=True)

    # Get rid of some outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]

    # Drop the rest of any nulls that may still be present
    df = df.dropna()

    # Calculate age of home from year built
    df.yearbuilt = 2017 - df.yearbuilt

    # Rename the columns for ease of use
    df.rename(columns={'taxvaluedollarcounty':'county_tax_value', 'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'square_feet', 'lotsizesquarefeet':'lot_size', 'buildingqualitytypeid':'building_quality', 'yearbuilt':'age', 'taxvaluedollarcnt':'tax_value', 'landtaxvaluedollarcnt':'land_tax_value', 'unitcnt':'unit_count', 'heatingorsystemdesc':'heating_system', 'structuretaxvaluedollarcnt':'structure_tax_value'}, inplace=True)

    # Create bins for age
    df['age_bin'] = pd.cut(df.age, bins=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], labels=['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-110', '110-120', '120-130', '130-140', '140-150'])

    # Create new feature for taxrate
    df['taxrate'] = (df.taxamount / df.tax_value) * 100

    # Create new feature for acres
    df['acres'] = df.lot_size / 43560

    # Bin the acres
    df['acres_bin'] = pd.cut(df.acres, bins=[0, 0.1, 0.15, 0.25, 0.5, 1, 5, 10, 20, 50, 200], labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Bin the tax value
    df['tax_value_bin'] = pd.cut(df.tax_value, bins=[0, 80000, 150000, 225000, 300000, 350000, 450000, 550000, 650000, 900000, 5000000], labels=['< $80,000', '$150,000', '$225,000', '$300,000', '$350,000', '$450,000', '$550,000', '$650,000', '$900,000', '$5,000,000'])

    # Bin the land tax value
    df['land_tax_value_bin'] = pd.cut(df.land_tax_value, bins=[0, 50000, 100000, 150000, 200000, 250000, 350000, 450000, 650000, 800000, 1000000], labels=['< $50,000', '$100,000', '$150,000', '$200,000', '$250,000', '$350,000', '$450,000', '$650,000', '$800,000', '$1,000,000'])

    # Bin the area sqft
    df['sqft_bin'] = pd.cut(df.square_feet, bins=[0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000], labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Dollars per square foot structure
    df['structure_dollar_per_sqft'] = df.structure_tax_value / df.square_feet
    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, bins=[0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 15000], labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Dollars per sqft of land
    df['land_dollar_per_sqft'] = df.land_tax_value / df.lot_size
    df['land_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins=[0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000], labels=['0', '1', '5-19', '20-49', '50-99', '100-249', '250-499', '500-999', '1000-1499', '1500-2000'])

    # Set data types of binned features
    df.astype({'sqft_bin':'float64', 'acres_bin':'float64', 'structure_dollar_sqft_bin':'float64'})

    # Create new feature for bedroom / bathroom ratio
    df['bath_bed_ratio'] = df.bathrooms / df.bedrooms

    # Dropping unneeded columns
    df = df.drop(columns=['parcelid', 'building_quality', 'county', 'lot_size', 'regionidcity', 'regionidcounty', 'regionidzip', 'roomcnt', 'unit_count', 'assessmentyear', 'transactiondate', 'heating_system'])

    # Drop the rest of any nulls that may still be present
    df = df.dropna()

    return df[((df.bathrooms <= 7) & (df.bedrooms <= 7) & (df.bathrooms >= 1) & (df.bedrooms >= 1) & (df.acres <= 20) & (df.square_feet <= 9000) & (df.taxrate <= 10))]



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def train_validate_test_split(df):
    train_and_validate, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_and_validate, train_size=0.75, random_state=123)
    return train, validate, test



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def scaled_data(train, validate, test):
    # Separate the target
    X_train = train.drop(columns=['logerror'])
    y_train = train['logerror']
    X_validate = validate.drop(columns=['logerror'])
    y_validate = validate['logerror']
    X_test = test.drop(columns=['logerror'])
    y_test = test['logerror']

    # Create the object
    scaler = MinMaxScaler()

    # Fit the object
    scaler.fit(X_train)

    # Use the object
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Make the target into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def select_kbest(X_train_scaled, y_train, k):
    '''
    Takes in predictors, target, and the number of features to select.
    Returns the names of the top k predictors.
    '''
    f_selector = SelectKBest(f_regression, k)
    f_selector = f_selector.fit(X_train_scaled, y_train)
    X_train_reduced = f_selector.transform(X_train_scaled)
    f_support = f_selector.get_support()
    f_feature = X_train_scaled.iloc[:, f_support].columns.tolist()
    return f_feature



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def rfe(X_train_scaled, y_train, k):
    '''
    Takes in predictors, target, and the number of features to select.
    Returns the names of the top k predictors.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, k)
    X_rfe = rfe.fit_transform(X_train_scaled, y_train)
    mask = rfe.support_
    rfe_feature = X_train_scaled.loc[:, mask].columns.tolist()
    return rfe_feature