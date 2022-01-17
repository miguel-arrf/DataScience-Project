import pandas


def apply_set1_imputation(data):
    for index, row in data.iterrows():
        if row["PERSON_TYPE"] == "Pedestrian":
            if pandas.isna(row["SAFETY_EQUIPMENT"]):
                data.at[index, "SAFETY_EQUIPMENT"] = "Does Not Apply"

            if pandas.isna(row["EJECTION"]):
                data.at[index, "EJECTION"] = "Unknown"

            data.at[index, "POSITION_IN_VEHICLE"] = "Does Not Apply"

            if pandas.isna(row["PED_ACTION"]):
                data.at[index, "PED_ACTION"] = "Unknown"

        else:
            if pandas.isna(row["SAFETY_EQUIPMENT"]):
                data.at[index, "SAFETY_EQUIPMENT"] = "Unknown"

            if row["PERSON_TYPE"] != "Occupant":
                data.at[index, "PED_LOCATION"] = "Does Not Apply"

            '''
            if row["PERSON_TYPE"] == "Occupant":
                if row["CONTRIBUTING_FACTOR_2"] != "Driver Inattention/Distraction":
                    data.at[index, "CONTRIBUTING_FACTOR_2"] = "Does Not Apply"
                if row["CONTRIBUTING_FACTOR_1"] != "Driver Inattention/Distraction":
                    data.at[index, "CONTRIBUTING_FACTOR_1"] = "Does Not Apply"
            else:
                data.at[index, "CONTRIBUTING_FACTOR_2"] = "Does Not Apply"
                data.at[index, "CONTRIBUTING_FACTOR_1"] = "Does Not Apply"
            '''
            if pandas.isna(row["CONTRIBUTING_FACTOR_2"]):
                data.at[index, "CONTRIBUTING_FACTOR_2"] = "Unknown"
            if pandas.isna(row["CONTRIBUTING_FACTOR_1"]):
                data.at[index, "CONTRIBUTING_FACTOR_1"] = "Unknown"

            if pandas.isna(row["EJECTION"]):
                data.at[index, "EJECTION"] = "Unknown"

            if pandas.isna(row["POSITION_IN_VEHICLE"]):
                data.at[index, "POSITION_IN_VEHICLE"] = "Unknown"

            if pandas.isna(row["PED_ACTION"]):
                data.at[index, "PED_ACTION"] = "Does Not Apply"

            if pandas.isna(row["PED_LOCATION"]):
                data.at[index, "PED_LOCATION"] = "Unknown"

        # if row["PED_LOCATION"] == "Does Not Apply":
        #     data.at[index, "PED_LOCATION"] = "Does Not Apply"

