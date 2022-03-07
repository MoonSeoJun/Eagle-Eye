import json


class DataCollector:
    def __init__(self):
        self.players_data_dict = {}
        self.players_data_arr = []
    
    def create_dataset(self, title):
        self.players_data_dict[title] = self.players_data_arr
        json_title = title.split('/')[-1].replace(".avi", "")
        with open(f"/Eagle-Eye/result/datas/{json_title}.json", "w") as json_file:
            json.dump(self.players_data_dict, json_file)