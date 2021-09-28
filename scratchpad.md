




def remove_outliers(df, k, col_list):
    '''
    Remove outliers from a list of columns.
    '''
    for col in col_list:
        # Get quartiles
        q1, q3 = df[col].quantile([.25, .75])
        
        # Get interquartile range
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        
        # Return the dataframe without the outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
        return df



def wrangle_zillow():
    # Get zillow data from csv or from database if there is no csv
    df = get_zillow_data()
    df = pd.DataFrame(df)

    # Nulls are rare for bedrooms and bathrooms
    # Nulls may not appear in train after splitting
    # Thus, I am dropping nulls for bedroom and bathroom now
    df.dropna()
    
    # Remove outliers from the dataframe
    df = remove_outliers(df, 2.0, ['calculatedfinishedsquarefeet','taxvaluedollarcnt', 'taxamount','bedroomcnt', 'bathroomcnt'] )
    
    # Lets get shorter and easier to understand names
    df = df.rename(columns= {'bedroomcnt' : 'bedrooms', 'bathroomcnt' : 'bathrooms', 'calculatedfinishedsquarefeet' : 'area_sqft', 'taxvaluedollarcnt' : 'tax_value', 'yearbuilt' : 'year_built', 'taxamount' : 'tax_amount'})

    # Replace a whitespace sequence or empty with a NaN value 
    # and reassign this manipulation to df.
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # fips appears to be a categorical number (i.e. like a zip code)
    # year_built may also be categorical if we aren't looking at age
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    return df



def wrangle_split(df):
    # Split the data into train, validate, and test subsets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(df, test_size=.3, random_state=123)
    
    # Need to fill nulls for: year_built, area_sqft, tax_value

    # Create -> Fit -> Use
    # Create the object
    imputer_year = SimpleImputer(strategy='most_frequent')

    # Fit the object to train
    imputer_year.fit(train[['year_built']])

    # Use the object to impute on train, validate, and test subsets
    train[['year_built']]=imputer_year.transform(train[['year_built']])
    validate[['year_built']]=imputer_year.transform(validate[['year_built']])
    test[['year_built']]=imputer_year.transform(test[['year_built']])

    # Now repeat the imputing process for area_sqft
    imputer_area = SimpleImputer(strategy='median')
    imputer_area.fit(train[['area_sqft']])

    train[['area_sqft']]=imputer_area.transform(train[['area_sqft']])
    validate[['area_sqft']]=imputer_area.transform(validate[['area_sqft']])
    test[['area_sqft']]=imputer_area.transform(test[['area_sqft']])

    # Now repeat the imputing process for tax_value
    imputer_tax = SimpleImputer(strategy='median')
    imputer_tax.fit(train[['tax_value']])

    train[['tax_value']]=imputer_tax.transform(train[['tax_value']])
    validate[['tax_value']]=imputer_tax.transform(validate[['tax_value']])
    test[['tax_value']]=imputer_tax.transform(test[['tax_value']])
    
    return train, validate, test



def scale_zillow(train, validate, test):
    # Create the object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the object
    # Scalers should only be fit to train to prevent data leakage
    scaler.fit(train)

    # Use the object
    # the same object that was fitted to train can be used on validate and test
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # For some reason, the minmax scaler converted the dataframes into series
    # and renamed the columns 0-6

    # The following lines convert those series back to dataframes 
    # and restore their column names

    train_scaled = pd.DataFrame(train_scaled)
    train_scaled = train_scaled.rename(columns={0:'bedrooms', 1:'bathrooms', 2:'area_sqft', 3:'tax_value', 4:'year_built',5:'tax_amount', 6:'fips'})

    validate_scaled = pd.DataFrame(validate_scaled)
    validate_scaled = validate_scaled.rename(columns={0:'bedrooms', 1:'bathrooms', 2:'area_sqft', 3:'tax_value', 4:'year_built',5:'tax_amount', 6:'fips'})

    test_scaled = pd.DataFrame(test_scaled)
    test_scaled = test_scaled.rename(columns={0:'bedrooms', 1:'bathrooms', 2:'area_sqft', 3:'tax_value', 4:'year_built',5:'tax_amount', 6:'fips'})

    return train_scaled, validate_scaled, test_scaled
