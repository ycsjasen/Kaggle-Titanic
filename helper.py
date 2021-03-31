import pandas as pd

def survive_df(df, attribute, cont):
    """
    Calculate total count of survivors and survival rate given an attribute existing in DataFrame and outputs
    a summary DataFrame

    Parameters:
        df : DataFrame
        attribute : string
                attribute to summarize
        cont : bool, default False
                If True, separates attribute values into 10 equal sized bins
    Returns:
        DataFrame
            A DataFrame of the calculated total passengers, total amount of survivors and the rate of
            survival for each category/bin of the attribute
    """
    if cont:
        df['Binned ' + attribute] = pd.cut(df[attribute], bins=10)
        binname = 'Binned ' + attribute
    else:
        binname = attribute
    total_count = df[[binname, 'Survived']].groupby([binname]).count()
    survive_count = df[[binname, 'Survived']].groupby([binname]).sum()

    total_count.rename(columns={'Survived': 'Total Passenger'}, inplace=True)
    survive_count.rename(columns={'Survived': 'Survivor Count'}, inplace=True)

    summary_df = pd.merge(total_count, survive_count, how='inner', on=binname)
    summary_df['Deceased count'] = summary_df['Total Passenger'] - summary_df['Survivor Count']
    summary_df['Survivor Rate'] = summary_df['Survivor Count'] / summary_df['Total Passenger']
    return summary_df
