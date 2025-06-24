

    def normalize_name(self, name):
        """
        Normalize names: remove spaces and capital letters.
        """
        if isinstance(name, str):
            return ''.join(name.upper().split())
        return ''


   def Build_Final_Dataframe(self):
        """
        It Creates a dataframe with the interest' variables, i.e. all columns for 
        our study. Also, it renames some columns.
        """
        cols_f = ['Main_ID', 'Spectype', 'Mo', 'Period', 'Eccentricity', 'Spin_period', 'Distance', 'Class']
        cols_n = ['Name', 'SpType', 'Mean_Mass', 'Teff', 'N_H', 'Max_Soft_Flux', 'Min_Soft_Flux', 'Max_Hard_Flux', 'Min_Hard_Flux']

        if not isinstance(self.common, list):
            self.common = [self.common]

        fortin_filtered = self.cat_fortin[self.cat_fortin['Main_ID'].isin(self.common)][cols_f]
        neuman_filtered = self.cat_neuman[self.cat_neuman['Name'].isin(self.common)][cols_n]
        fortin_filtered = fortin_filtered.rename(columns={'Main_ID': 'Object'})
        neuman_filtered = neuman_filtered.rename(columns={'Name': 'Object'})

        final_dataframe = pd.merge(fortin_filtered, neuman_filtered, on='Object', how='outer')

        final_dataframe['Mean_Soft_Flux'] = np.sqrt(final_dataframe['Min_Soft_Flux'] * final_dataframe['Max_Soft_Flux'])
        final_dataframe['Mean_Hard_Flux'] = np.sqrt(final_dataframe['Min_Hard_Flux'] * final_dataframe['Max_Hard_Flux'])
        final_dataframe['Hardness'] = final_dataframe['Mean_Hard_Flux'] / final_dataframe['Mean_Soft_Flux']

        final_dataframe.rename(columns={
            "SpType": "SpType_Fortin", 
            "Spectype": "SpType_Neumann", 
            "Mean_Mass": "M_X", 
            "Mo": "M*"
        }, inplace=True)

        return final_dataframe

