import pandas


def apply_set1_imputation(data):
    for index, row in data.iterrows():
        if row["PERSON_TYPE"] == "Pedestrian":

            data.at[index, "SAFETY_EQUIPMENT"] = "NotApplicable"
            data.at[index, "EJECTION"] = "NotApplicable"
            data.at[index, "POSITION_IN_VEHICLE"] = "NotApplicable"

            if pandas.isna(row["PED_ACTION"]):
                data.at[index, "PED_ACTION"] = "Unknown"

        else:

            if pandas.isna(row["SAFETY_EQUIPMENT"]):
                data.at[index, "SAFETY_EQUIPMENT"] = "Unknown"

            if row["PERSON_TYPE"] != "Occupant":
                data.at[index, "PED_LOCATION"] = "NotApplicable"

            if row["PERSON_TYPE"] == "Occupant":
                if row["CONTRIBUTING_FACTOR_2"] != "Driver Inattention/Distraction":
                    data.at[index, "CONTRIBUTING_FACTOR_2"] = "NotApplicable"
                if row["CONTRIBUTING_FACTOR_1"] != "Driver Inattention/Distraction":
                    data.at[index, "CONTRIBUTING_FACTOR_1"] = "NotApplicable"
            else:
                data.at[index, "CONTRIBUTING_FACTOR_2"] = "NotApplicable"
                data.at[index, "CONTRIBUTING_FACTOR_1"] = "NotApplicable"

            if pandas.isna(row["EJECTION"]):
                data.at[index, "EJECTION"] = "Unknown"

            if pandas.isna(row["POSITION_IN_VEHICLE"]):
                data.at[index, "POSITION_IN_VEHICLE"] = "Unknown"

            if pandas.isna(row["PED_ACTION"]):
                data.at[index, "PED_ACTION"] = "NotApplicable"

            if pandas.isna(row["PED_LOCATION"]):
                data.at[index, "PED_LOCATION"] = "NotApplicable"

        if pandas.isna(row["VEHICLE_ID"]):
            data.at[index, "VEHICLE_ID"] = -1