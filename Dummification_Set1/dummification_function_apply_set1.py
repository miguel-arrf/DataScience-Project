def apply_set1_imputation(data, columns, type_of_encoder):
    encoder = type_of_encoder(cols=columns, verbose=False)
    columns = []

    for column in data.columns:
        if column != "PERSON_INJURY":
            columns.append(column)

    return encoder.fit_transform(data[columns], data["PERSON_INJURY"])
