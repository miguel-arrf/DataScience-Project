import xlrd
from pandas import read_csv

filename = '../data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)
data.drop(data[(data.PERSON_AGE < 0) | (data.PERSON_AGE > 200)].index,inplace=True)
data.drop(data.loc[data['PERSON_SEX'] == "U"].index, inplace=True)

def encode_EJECTION_LOCATION_ROLE_TYPE(dataset):
    encode_ejection = [("Not Ejected", 5), ("Trapped", 7), ("Ejected", 10), ("Partially Ejected", 8), ("Unknown", 0)]
    encode_location = {"Does Not Apply": 3, "Pedestrian/Bicyclist/Other Pedestrian Not at Intersection": 4,
                       "Pedestrian/Bicyclist/Other Pedestrian at Intersection": 5, "Unknown": 1}
    encode_role = {"Driver": 5, "Pedestrian": 4, "Passenger": 3, "Other": 1, "In-Line Skater": 2}
    encode_type = {"Occupant": 2, "Pedestrian": 3, "Bicyclist": 1, "Other Motorized": 4}

    for value in encode_ejection:
        dataset["EJECTION"].loc[(dataset["EJECTION"] == value[0])] = value[1]

    for key in encode_location:
        dataset["PED_LOCATION"].loc[(dataset["PED_LOCATION"] == key)] = encode_location[key]

    for value in encode_role:
        dataset["PED_ROLE"].loc[(dataset["PED_ROLE"] == value)] = encode_role[value]

    for value in encode_type:
        dataset["PERSON_TYPE"].loc[(dataset["PERSON_TYPE"] == value)] = encode_type[value]


def encode(dataset):
    dataset_copy = dataset.copy()

    workbook = xlrd.open_workbook(f'encoding.xlsx', on_demand=True)
    sheetNames = workbook.sheet_names()
    encoding = []
    for sheet in range(len(sheetNames)):
        worksheet = workbook.sheet_by_index(sheet)

        first_row = []  # The row where we stock the name of the column
        for col in range(worksheet.ncols):
            first_row.append(worksheet.cell_value(0, col))
        # transform the workbook to a list of dictionaries
        data = []
        for row in range(1, worksheet.nrows):
            elm = {}
            for col in range(worksheet.ncols):
                elm[first_row[col]] = worksheet.cell_value(row, col)
            data.append(elm)
        encoding.append(data)

        for value in data:
            print(value)
            dataset_copy[sheetNames[sheet]].loc[(dataset_copy[sheetNames[sheet]] == value['Name'])] = value['Group']

    #encode_EJECTION_LOCATION_ROLE_TYPE(dataset_copy)

    # for symbolic_var in ["CRASH_TIME", "CRASH_DATE"]:
    #    dataset_copy = encode_by_order(dataset_copy, symbolic_var)

    return dataset_copy


if __name__ == '__main__':
    print("Olá D:")
    data = encode(data)
    data.to_csv("encodedWithGranularity.csv")