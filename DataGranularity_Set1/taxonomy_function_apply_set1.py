import pandas


class taxonomy_set1:

    def apply_set1_taxonomy(self, data):

        def transform_equipment(v):
            aux = ""
            if "nan" in str(v) or "None" in str(v) or "Unknown" in str(v):
                aux = "NOT EQUIPPED"
            else:
                if "Helmet" in str(v):
                    aux += "HEAD & "
                if "Belt" in str(v) or "Harness" in str(v):
                    aux += "UPPER BODY & "
                if "Pads" in str(v) or "Stoppers" in str(v):
                    aux += "LOWER BODY & "
                if "Air Bag" in str(v):
                    aux += "AIRBAG & "
                if "Child Restraint" in str(v):
                    aux += "CHILD RESTRAINT & "
                if "Other" in str(v):
                    aux += "OTHER"
            if aux[-3:] == " & ":
                aux = aux[:-3]
            v = aux
            return v

        def transform_injury(v):
            aux = ""
            if "Does Not Apply" in str(v) or "Unknown" in str(v):
                aux = "NONE"
            if "Leg" in str(v):
                aux = "LOWER BODY"
            if "Eye" in str(v) or "Head" in str(v) or "Face" in str(v) or "Neck" in str(v):
                aux = "HEAD"
            if "Arm" in str(v) or "Back" in str(v) or "Chest" in str(v) or "Abdomen" in str(v):
                aux = "UPPER BODY"
            if "Entire Body" in str(v):
                aux = "ENTIRE BODY"
            v = aux
            return v

        data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].apply(lambda x: transform_equipment(x))
        data['BODILY_INJURY'] = data['BODILY_INJURY'].apply(lambda x: transform_injury(x))

        injury_order = ["NONE", "LOWER BODY", "UPPER BODY", "HEAD", "ENTIRE BODY"]
        equipment_order = ["NOT EQUIPPED", "UPPER BODY", "LOWER BODY", "HEAD", "AIRBAG", "CHILD RESTRAINT", "OTHER",
                           "HEAD & OTHER", "UPPER BODY & AIRBAG", "AIRBAG & CHILD RESTRAINT"]

        c_classes = pandas.api.types.CategoricalDtype(ordered=True, categories=equipment_order)
        data['SAFETY_EQUIPMENT'] = data['SAFETY_EQUIPMENT'].astype(c_classes)

        c_classes2 = pandas.api.types.CategoricalDtype(ordered=True, categories=injury_order)
        data['BODILY_INJURY'] = data['BODILY_INJURY'].astype(c_classes2)
