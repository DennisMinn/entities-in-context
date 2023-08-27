# -*- coding: utf-8 -*-
"""Pre-process data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zMIjUrkmB0CwoH9alxHpCxg2HJcNlsqw
"""

import json

# Read train and valid data
train_data = []
with open('../../../data/clutrr/train.jsonl') as f:
    for line in f:
        train_data.append(json.loads(line))

valid_data = []
with open('../../../data/clutrr/valid.jsonl') as f:
    for line in f:
        valid_data.append(json.loads(line))

print(f'Length of train data: {len(train_data)}')
print(f'Length of validation data: {len(valid_data)}')

# Merging the train and validaiton data to increase the set from which the demonstraitons are sampled
total_data = train_data + valid_data
train_data = total_data[:len(total_data)-1000]
valid_data = total_data[-1000:]

print(f'Length of revised train data: {len(train_data)}')
print(f'Length of revised validation data: {len(valid_data)}')

# Method to find genders of names using genderize.io
import requests
def find_genders(data):
    gender_dict = {}
    for i in range(len(data)):
        query_1 = str(data[i]['query'][1])
        api_call = 'https://api.genderize.io/?name=' + data[i]['name_map'][query_1]
        response = requests.get(api_call)
        gender_dict[data[i]['name_map'][query_1]] = response.json()['gender']
    return gender_dict

# Genders after using the above find_genders method on train data
# i.e. train_data_gender = find_genders(train_data)
train_data_gender = {'Ida': 'female',
 'Augusta': 'female',
 'Byron': 'male',
 'Hester': 'female',
 'Doss': 'male',
 'Gertrude': 'female',
 'Colin': 'male',
 'Tobe': 'male',
 'Aron': 'male',
 'Sula': 'female',
 'Leota': 'female',
 'Gene': 'male',
 'Pablo': 'male',
 'Una': 'female',
 'Anderson': 'male',
 'Joella': 'female',
 'Isam': 'male',
 'Dock': 'male',
 'Lucius': 'male',
 'Melva': 'female',
 'Georgiana': 'female',
 'Lavonia': 'female',
 'Alys': 'female',
 'Josephine': 'female',
 'Allie': 'female',
 'Andrew': 'male',
 'Nolia': 'female',
 'Nelia': 'female',
 'Laura': 'female',
 'Alford': 'male',
 'Hattie': 'female',
 'Author': 'male',
 'Noel': 'male',
 'Clinton': 'male',
 'Glen': 'male',
 'Doctor': 'male',
 'Claud': 'male',
 'Hermine': 'female',
 'Garfield': 'male',
 'Theo': 'male',
 'Caesar': 'male',
 'Olive': 'female',
 'Lorena': 'female',
 'Cleveland': 'male',
 'Stacy': 'female',
 'Cliff': 'male',
 'Sannie': 'female',
 'Vernie': 'female',
 'Tony': 'male',
 'Nena': 'female',
 'Pollie': 'female',
 'Elisa': 'female',
 'Celie': 'female',
 'Austin': 'male',
 'Norah': 'female',
 'Alverta': 'female',
 'Yee': 'female',
 'Fay': 'female',
 'Delos': 'male',
 'Lyda': 'female',
 'Meda': 'female',
 'Jonas': 'male',
 'Harris': 'male',
 'Garry': 'male',
 'Guy': 'male',
 'Marilla': 'female',
 'Maurice': 'male',
 'Emory': 'male',
 'Ludie': 'female',
 'Gena': 'female',
 'Estelle': 'female',
 'Dallas': 'male',
 'Jack': 'male',
 'Egbert': 'male',
 'Helen': 'female',
 'Ezekiel': 'male',
 'Hulda': 'female',
 'Winfield': 'male',
 'Rutherford': 'male',
 'Hannah': 'female',
 'Lucie': 'female',
 'Katy': 'female',
 'Pete': 'male',
 'Agatha': 'female',
 'Doshie': 'female',
 'Thelma': 'female',
 'Louie': 'male',
 'Edith': 'female',
 'Rosalia': 'female',
 'Dora': 'female',
 'Horace': 'male',
 'Tella': 'female',
 'Travis': 'male',
 'Pat': 'male',
 'Gracie': 'female',
 'Ishmael': 'male',
 'Mayme': 'female',
 'Delphia': 'female',
 'Shelton': 'male',
 'Claudie': 'female',
 'Ferdinand': 'male',
 'Walter': 'male',
 'Janet': 'female',
 'Selena': 'female',
 'Tressa': 'female',
 'Sherman': 'male',
 'Judith': 'female',
 'Mame': 'female',
 'Lonzo': 'male',
 'Terry': 'male',
 'Roland': 'male',
 'Huey': 'male',
 'Ambrose': 'male',
 'Aimee': 'female',
 'Sally': 'female',
 'Flora': 'female',
 'Drew': 'male',
 'Selma': 'female',
 'Sydney': 'female',
 'Math': 'male',
 'Ella': 'female',
 'Netta': 'female',
 'Liller': 'female',
 'Paul': 'male',
 'Fannie': 'female',
 'Orville': 'male',
 'Washington': 'male',
 'Elmo': 'male',
 'Clifford': 'male',
 'Vada': 'female',
 'Gladys': 'female',
 'Kattie': 'female',
 'Clemie': 'female',
 'Lutie': 'female',
 'Clement': 'male',
 'Fred': 'male',
 'Julia': 'female',
 'Bert': 'male',
 'Mahalia': 'female',
 'Fredrick': 'male',
 'Lucina': 'female',
 'Orren': 'male',
 'Billy': 'male',
 'Nan': 'female',
 'Ballard': 'male',
 'Leora': 'female',
 'Emmaline': 'female',
 'Frieda': 'female',
 'Tomas': 'male',
 'Therese': 'female',
 'Emanuel': 'male',
 'Hazel': 'female',
 'Merle': 'female',
 'Ola': 'female',
 'Maude': 'female',
 'Alden': 'male',
 'Daisie': 'female',
 'Green': 'male',
 'Zella': 'female',
 'Fronie': 'female',
 'Alpha': 'male',
 'Dana': 'female',
 'Rollin': 'male',
 'Emma': 'female',
 'Urban': 'male',
 'Gilbert': 'male',
 'Evelyn': 'female',
 'Erasmus': 'male',
 'Indiana': 'female',
 'Margaret': 'female',
 'Letta': 'female',
 'Omie': 'male',
 'Myron': 'male',
 'Isham': 'male',
 'Kizzie': 'female',
 'Hollie': 'female',
 'Vera': 'female',
 'Celina': 'female',
 'Eliza': 'female',
 'Ottilia': 'female',
 'Layton': 'male',
 'Taylor': 'female',
 'Sara': 'female',
 'Rafael': 'male',
 'Justice': 'male',
 'Lucy': 'female',
 'Melvina': 'female',
 'Godfrey': 'male',
 'Eula': 'female',
 'Zack': 'male',
 'Nat': 'male',
 'Carrie': 'female',
 'Jimmie': 'male',
 'Boston': 'male',
 'Frankie': 'male',
 'Jonathan': 'male',
 'Edmund': 'male',
 'Crawford': 'male',
 'Amie': 'female',
 'Tom': 'male',
 'Rosia': 'female',
 'Guss': 'male',
 'Norris': 'male',
 'Malachi': 'male',
 'Malissa': 'female',
 'Levi': 'male',
 'Jean': 'female',
 'Warner': 'male',
 'Mike': 'male',
 'Albion': 'male',
 'Ben': 'male',
 'Natalie': 'female',
 'Cyril': 'male',
 'Melville': 'male',
 'Edythe': 'female',
 'Marian': 'female',
 'Rolland': 'male',
 'Erie': 'male',
 'Harriet': 'female',
 'Easter': 'female',
 'Mai': 'female',
 'Wilfred': 'male',
 'Arizona': 'female',
 'Edna': 'female',
 'Allen': 'male',
 'Gust': 'male',
 'Willian': 'male'}

# Genders after using the above find_genders method on validation data
# i.e. valid_data_gender = find_genders(valid_data)
valid_data_gender = {'Mima': 'female',
 'Pearlie': 'female',
 'Martin': 'male',
 'Pearl': 'female',
 'Viola': 'female',
 'Annie': 'female',
 'Abe': 'male',
 'Paulina': 'female',
 'Hattie': 'female',
 'Webb': 'male',
 'Jennie': 'female',
 'Alphonse': 'male',
 'Onie': 'female',
 'Rose': 'female',
 'Lucia': 'female',
 'Enrique': 'male',
 'Adah': 'female',
 'Leonie': 'female',
 'Davie': 'male',
 'Theodora': 'female',
 'Grayce': 'female',
 'Alonza': 'male',
 'Ila': 'female',
 'Mattie': 'female',
 'Rosella': 'female',
 'Nola': 'female',
 'Glenn': 'male',
 'Dolly': 'female',
 'Alvin': 'male',
 'Lucy': 'female',
 'Adline': 'female',
 'Tella': 'female',
 'Corrie': 'female',
 'Bernard': 'male',
 'Joeseph': 'male',
 'Adella': 'female',
 'Eric': 'male',
 'Lyman': 'male',
 'Wilhelmine': 'female',
 'Leora': 'female',
 'Major': 'male',
 'Burr': 'male',
 'Harold': 'male',
 'Angela': 'female',
 'Vern': 'male',
 'Zack': 'male',
 'Neppie': 'female',
 'Mintie': 'female',
 'Queen': 'female',
 'Lular': 'female',
 'Theo': 'male',
 'Winona': 'female',
 'Carlotta': 'female',
 'Eben': 'male',
 'Diana': 'female',
 'Mathew': 'male',
 'Zula': 'female',
 'Faith': 'female',
 'Hannah': 'female',
 'Bessie': 'female',
 'Melinda': 'female',
 'Evie': 'female',
 'Hulda': 'female',
 'Noel': 'male',
 'Donie': 'male',
 'Paralee': 'female',
 'Juanita': 'female',
 'Clara': 'female',
 'Burley': 'male',
 'Retta': 'female',
 'Tilden': 'male',
 'Warner': 'male',
 'Timothy': 'male',
 'Queenie': 'female',
 'Gottlieb': 'male',
 'Love': 'female',
 'Stonewall': 'male',
 'Nealie': 'female',
 'Newton': 'male',
 'Neva': 'female',
 'Flossie': 'female',
 'Gilbert': 'male',
 'Edith': 'female',
 'Eva': 'female',
 'Ludwig': 'male',
 'Miles': 'male',
 'Corine': 'female',
 'Delia': 'female',
 'Icy': 'female',
 'Everett': 'male',
 'Asa': 'male',
 'Arthur': 'male',
 'Madie': 'female',
 'Gorge': 'male',
 'Hector': 'male',
 'Harrison': 'male',
 'Ottilie': 'female',
 'Rollin': 'male',
 'Victoria': 'female',
 'Scott': 'male',
 'Alonzo': 'male',
 'Robert': 'male',
 'Elena': 'female',
 'Leanna': 'female',
 'Lina': 'female',
 'Wilmer': 'male',
 'Louisa': 'female',
 'Elsie': 'female',
 'Lenora': 'female',
 'Mathias': 'male',
 'Lillie': 'female',
 'Ana': 'female',
 'Eve': 'female',
 'Archibald': 'male',
 'Debbie': 'female',
 'Era': 'female',
 'Baxter': 'male',
 'Claud': 'male',
 'June': 'female',
 'Obed': 'male',
 'Jesse': 'male',
 'Rita': 'female',
 'Celestine': 'female',
 'Elmo': 'male',
 'Lavinia': 'female',
 'Art': 'male',
 'Anthony': 'male',
 'Armand': 'male',
 'Loma': 'female',
 'Nick': 'male',
 'Isaiah': 'male',
 'Rosa': 'female',
 'Winfield': 'male',
 'Mack': 'male',
 'Emelie': 'female',
 'Dillard': 'male',
 'Sadie': 'female',
 'Polly': 'female',
 'Watson': 'male',
 'Eddie': 'male',
 'Baldwin': 'male',
 'Isabelle': 'female',
 'Lonie': 'female',
 'Dolores': 'female',
 'Grace': 'female',
 'Maud': 'female',
 'Margaret': 'female',
 'Dudley': 'male',
 'Emelia': 'female',
 'Millicent': 'female',
 'Abbie': 'female',
 'Wash': 'male',
 'Celia': 'female',
 'Emmer': 'male',
 'Malinda': 'female',
 'Gertie': 'female',
 'Brad': 'male',
 'Theodore': 'male',
 'Frona': 'female',
 'Dave': 'male',
 'Hershel': 'male',
 'Bert': 'male',
 'Liddie': 'female',
 'Suzanne': 'female',
 'Shelton': 'male',
 'Omar': 'male',
 'Lyda': 'female',
 'Winifred': 'female',
 'Mina': 'female',
 'Orlando': 'male',
 'Drucilla': 'female',
 'Mart': 'male',
 'Eugenia': 'female',
 'Junious': 'male',
 'Elmore': 'male',
 'Lafe': 'male',
 'Minda': 'female',
 'Willard': 'male',
 'Gwendolyn': 'female',
 'Edson': 'male',
 'Mable': 'female',
 'Rubin': 'male',
 'Phil': 'male',
 'Bryant': 'male',
 'Tressa': 'female',
 'James': 'male',
 'Lanie': 'female',
 'Eli': 'male',
 'Nellie': 'female',
 'Gracia': 'female',
 'Ephraim': 'male',
 'Arvilla': 'female',
 'Monica': 'female',
 'Nannie': 'female',
 'Florida': 'female',
 'Callie': 'female',
 'Bennie': 'male',
 'Johannah': 'female',
 'Cinda': 'female',
 'Zilpha': 'female',
 'Augustine': 'male',
 'Ava': 'female',
 'Jessie': 'female',
 'Jessica': 'female',
 'Barney': 'male',
 'Adrienne': 'female',
 'Byron': 'male',
 'Araminta': 'female',
 'Harve': 'male',
 'Lewis': 'male',
 'Lou': 'female',
 'Dellar': 'female',
 'Clint': 'male',
 'Donald': 'male',
 'Jimmie': 'male',
 'Benjamine': 'male',
 'Leander': 'male',
 'Clemmie': 'female',
 'Bertha': 'female',
 'Lena': 'female',
 'Freddie': 'male',
 'Mazie': 'female',
 'Juan': 'male',
 'Giles': 'male',
 'Vance': 'male',
 'Addison': 'male',
 'Wilbert': 'male',
 'Lacy': 'female',
 'Gary': 'male',
 'Russel': 'male',
 'Marie': 'female',
 'John': 'male',
 'Vincent': 'male',
 'Carolyn': 'female',
 'Oliver': 'male',
 'Aubrey': 'female',
 'Ollie': 'male',
 'Jess': 'male',
 'Sydney': 'female',
 'Joshua': 'male',
 'Caesar': 'male',
 'Elbert': 'male',
 'Vivian': 'female',
 'Alois': 'male',
 'Theodosia': 'female',
 'Della': 'female',
 'Alphonso': 'male',
 'Veronica': 'female',
 'Zelda': 'female',
 'Louie': 'male',
 'Amos': 'male',
 'Merton': 'male',
 'Becky': 'female',
 'Emory': 'male',
 'Ashby': 'male',
 'Yee': 'female',
 'Sina': 'female',
 'Pauline': 'female',
 'Margarette': 'female',
 'Isham': 'male',
 'Johnson': 'male',
 'Lea': 'female',
 'Elroy': 'male',
 'Harriet': 'female',
 'Euphemia': 'female',
 'Priscilla': 'female',
 'Hunter': 'male',
 'Hettie': 'female',
 'Shirley': 'female',
 'Mabel': 'female',
 'Lollie': 'female',
 'Cyrus': 'male',
 'Dinah': 'female',
 'Juliet': 'female',
 'Earnest': 'male',
 'Eola': 'female',
 'Thurman': 'male',
 'Elsa': 'female',
 'Vinnie': 'male',
 'Marcella': 'female',
 'Rosalee': 'female',
 'Emmie': 'female',
 'Vannie': 'female',
 'Adelle': 'female',
 'Emile': 'male',
 'Dell': 'male',
 'Wesley': 'male',
 'George': 'male',
 'Colonel': 'male',
 'Lutie': 'female',
 'Aaron': 'male',
 'Malissa': 'female',
 'Rae': 'female',
 'Florence': 'female',
 'Stanley': 'male',
 'Gordon': 'male',
 'Alva': 'female',
 'Christine': 'female',
 'Hazel': 'female',
 'Denver': 'male',
 'Dock': 'male',
 'Beatrice': 'female',
 'Augusta': 'female',
 'Kathleen': 'female',
 'Esau': 'male',
 'Addie': 'female',
 'Altha': 'female',
 'Mena': 'female',
 'Ednah': 'female',
 'Isreal': 'male',
 'Len': 'male',
 'Harley': 'male',
 'Sumner': 'male',
 'Eugene': 'male',
 'Olen': 'male',
 'Bart': 'male',
 'Loretta': 'female',
 'Aurthur': 'male',
 'Llewellyn': 'male',
 'Robbie': 'male',
 'Zena': 'female',
 'Buck': 'male',
 'Drew': 'male',
 'Mammie': 'female',
 'Cordie': 'female',
 'Hilma': 'female',
 'Jarrett': 'male',
 'Irven': 'male',
 'Edgar': 'male',
 'Adam': 'male',
 'Rafe': 'male',
 'Izora': 'female',
 'Hoyt': 'male',
 'Lucien': 'male',
 'Soloman': 'male',
 'Gustav': 'male',
 'Kitty': 'female',
 'Ples': 'male',
 'Alice': 'female',
 'Winnifred': 'female',
 'Elizabeth': 'female',
 'Jacob': 'male',
 'Issac': 'male',
 'Amy': 'female',
 'Babe': 'female',
 'Joan': 'female',
 'Ena': 'female',
 'Merritt': 'male',
 'Holly': 'female',
 'Tennessee': 'female',
 'Oda': 'female',
 'Felicia': 'female',
 'Fronie': 'female',
 'Idella': 'female',
 'Stella': 'female',
 'Percival': 'male',
 'Flora': 'female',
 'Thad': 'male',
 'Cecelia': 'female',
 'Felipe': 'male',
 'Clementine': 'female',
 'Fidelia': 'female',
 'Malvina': 'female',
 'Lola': 'female',
 'Calla': 'female',
 'Finis': 'male',
 'Ethyl': 'female',
 'Berry': 'male',
 'Abbott': 'male',
 'Harl': 'male',
 'Teresa': 'female',
 'Pinkie': 'female',
 'Harvey': 'male',
 'Wilda': 'female',
 'Algie': 'male',
 'Dena': 'female',
 'Rhoda': 'female',
 'Benjamin': 'male',
 'Merle': 'female',
 'Margery': 'female',
 'Ray': 'male',
 'Ellar': 'female',
 'Hollis': 'male',
 'Earl': 'male',
 'Pink': 'female',
 'Eldridge': 'male',
 'Obie': 'male',
 'Dixie': 'female',
 'Marion': 'female',
 'Kathryn': 'female',
 'Washington': 'male',
 'Amie': 'female',
 'Halsey': 'male',
 'Hyrum': 'male',
 'Wylie': 'male',
 'Lonnie': 'male',
 'Henry': 'male',
 'Matthew': 'male',
 'Claire': 'female',
 'Price': 'male',
 'Ina': 'female',
 'Cassius': 'male',
 'Darius': 'male',
 'Linnie': 'female',
 'Ione': 'female',
 'Clair': 'female',
 'Minta': 'female',
 'Orson': 'male',
 'Molly': 'female',
 'Blanche': 'female',
 'Georgia': 'female',
 'Uriah': 'male',
 'Vida': 'female',
 'Maurice': 'male',
 'Vernie': 'female',
 'Tina': 'female',
 'Amelia': 'female',
 'Kattie': 'female',
 'Bird': 'male',
 'Maryann': 'female',
 'Helena': 'female',
 'Leona': 'female',
 'Budd': 'male',
 'Myrtle': 'female',
 'Winnie': 'female',
 'Garland': 'male',
 'Bailey': 'female',
 'Magdalena': 'female',
 'Mai': 'female',
 'Enos': 'male',
 'Louvenia': 'female',
 'Ida': 'female',
 'Murray': 'male',
 'Richmond': 'male',
 'Effie': 'female',
 'Monroe': 'male',
 'Orie': 'female',
 'Eula': 'female',
 'Eunice': 'female',
 'Elmer': 'male',
 'Wellington': 'male',
 'Dillie': 'female',
 'Zella': 'female',
 'Tempie': 'female',
 'Osie': 'male',
 'Parthenia': 'female',
 'Jean': 'female',
 'Anson': 'male',
 'Carolina': 'female',
 'Clem': 'male',
 'Riley': 'male',
 'Haywood': 'male',
 'Orville': 'male',
 'Pat': 'male',
 'Abraham': 'male',
 'Lura': 'female',
 'Hezekiah': 'male',
 'Elzie': 'female',
 'Justice': 'male',
 'Vena': 'female',
 'Lillis': 'female',
 'Daisey': 'female',
 'Phyllis': 'female',
 'Elma': 'female',
 'Martha': 'female',
 'Kirk': 'male',
 'Jeremiah': 'male',
 'Jewel': 'female',
 'Otha': 'male',
 'Emilia': 'female',
 'Mortimer': 'male',
 'Sid': 'male',
 'Alba': 'female',
 'Burl': 'male',
 'Rice': 'male',
 'Solon': 'male',
 'Lon': 'male',
 'Green': 'male',
 'Cato': 'male',
 'Genie': 'female',
 'Lavenia': 'female',
 'Lucille': 'female',
 'Smith': 'male',
 'Genevieve': 'female',
 'Estell': 'female',
 'Duke': 'male',
 'Boyd': 'male',
 'David': 'male',
 'Rube': 'male',
 'Rilla': 'female',
 'Isabell': 'female',
 'Doshie': 'female',
 'Nevada': 'male',
 'Ellsworth': 'male',
 'Hermine': 'female',
 'William': 'male',
 'Mame': 'female',
 'Logan': 'male',
 'Jewell': 'female',
 'Geraldine': 'female',
 'Magnolia': 'female',
 'Zeke': 'male',
 'May': 'female',
 'Dorothy': 'female',
 'Anita': 'female',
 'Joanna': 'female',
 'Myron': 'male',
 'Norris': 'male',
 'Hollie': 'female',
 'Clarence': 'male',
 'Nina': 'female',
 'Bennett': 'male',
 'Huey': 'male',
 'Dessa': 'female',
 'Doc': 'male',
 'Sylvester': 'male',
 'Eudora': 'female',
 'Albert': 'male',
 'Mitchel': 'male',
 'Lavina': 'female',
 'Tennie': 'female',
 'Aron': 'male',
 'Abby': 'female',
 'Kathryne': 'female',
 'Finley': 'male',
 'Kate': 'female',
 'Roberta': 'female',
 'Marjorie': 'female',
 'Andre': 'male',
 'Claudine': 'female',
 'Nora': 'female',
 'Hardie': 'male',
 'Dewitt': 'male',
 'Lavern': 'female',
 'Evelina': 'female',
 'Valentine': 'female',
 'Clifford': 'male',
 'Lynn': 'female',
 'Delphia': 'female',
 'Lenard': 'male',
 'Hester': 'female',
 'Betty': 'female',
 'Geo': 'male',
 'Lem': 'male',
 'Perley': 'female',
 'Burton': 'male',
 'Adolphus': 'male',
 'Lucie': 'female',
 'Leola': 'female',
 'Adda': 'female',
 'Curt': 'male',
 'Alys': 'female',
 'Omer': 'male',
 'Christena': 'female',
 'Cal': 'male',
 'Cora': 'female',
 'Lizzie': 'female',
 'Alma': 'female',
 'Odessa': 'female',
 'Bama': 'female',
 'Saul': 'male',
 'Erma': 'female',
 'Meta': 'female',
 'Urban': 'male',
 'Hardin': 'male',
 'Gee': 'male',
 'Wilson': 'male',
 'Julian': 'male',
 'Dorsey': 'male',
 'Arvid': 'male',
 'Otto': 'male',
 'Paul': 'male',
 'Manerva': 'female',
 'Al': 'male',
 'Goldie': 'female',
 'Russell': 'male',
 'Roland': 'male',
 'Walter': 'male',
 'Aline': 'female',
 'Morton': 'male',
 'Iva': 'female',
 'Jordan': 'male',
 'Blanch': 'female',
 'Creed': 'male',
 'Osa': 'male',
 'Reba': 'female',
 'Isaac': 'male',
 'Clarissa': 'female',
 'Grover': 'male',
 'Salome': 'female',
 'Fernando': 'male',
 'Lulu': 'female',
 'Williams': 'male',
 'Theda': 'female',
 'Gertrude': 'female',
 'Minnie': 'female',
 'Letitia': 'female',
 'Cruz': 'male',
 'Bena': 'female',
 'Chesley': 'male',
 'Ula': 'female',
 'Travis': 'male',
 'Ezekiel': 'male',
 'Oran': 'male',
 'Esta': 'female',
 'Chloe': 'female',
 'Cleveland': 'male',
 'Elise': 'female',
 'Joe': 'male',
 'Jack': 'male',
 'Felix': 'male',
 'Hermann': 'male',
 'Dennis': 'male',
 'Ford': 'male',
 'Ardelia': 'female',
 'Philomena': 'female',
 'Etha': 'female',
 'Jeanne': 'female',
 'Albertine': 'female',
 'Janet': 'female',
 'Louis': 'male',
 'Nona': 'female',
 'Camille': 'female',
 'Dwight': 'male',
 'Fred': 'male',
 'Arlie': 'male',
 'Alla': 'female',
 'Eliza': 'female',
 'Minervia': 'female',
 'Marvin': 'male',
 'Sabina': 'female',
 'Verona': 'female',
 'Denis': 'male',
 'Esther': 'female',
 'Lulie': 'female',
 'Press': 'male',
 'Conrad': 'male',
 'Leone': 'male',
 'Naomi': 'female',
 'Benjiman': 'male',
 'Wm': 'male',
 'Lafayette': 'male',
 'Elick': 'male',
 'Zona': 'female',
 'Preston': 'male',
 'Mallie': 'female',
 'Anastacio': 'male',
 'Garfield': 'male',
 'Willa': 'female',
 'Mahala': 'female',
 'Egbert': 'male',
 'Juana': 'female',
 'Josephine': 'female',
 'Clarance': 'male',
 'Otelia': 'female',
 'Isam': 'male',
 'Mittie': 'female',
 'Clare': 'female',
 'General': 'male',
 'Maye': 'female',
 'Julie': 'female',
 'Amalia': 'female',
 'Peter': 'male',
 'Helene': 'female',
 'Cordia': 'female',
 'Fate': 'male',
 'Tim': 'male',
 'Kizzie': 'female',
 'Morris': 'male',
 'Ottis': 'male',
 'Ruby': 'female',
 'Evan': 'male',
 'Augustus': 'male',
 'Christina': 'female',
 'Sidney': 'male',
 'Andrew': 'male',
 'Patty': 'female',
 'Dona': 'female',
 'Pearle': 'female',
 'Judy': 'female',
 'Ervin': 'male',
 'Lige': 'male',
 'Charles': 'male',
 'Mozella': 'female',
 'Lucina': 'female',
 'Rosalie': 'female',
 'Cassie': 'female',
 'Elvina': 'female',
 'Josiephine': 'female',
 'Minor': 'male',
 'Delphine': 'female',
 'Sue': 'female',
 'Adrian': 'male',
 'Clay': 'male',
 'Huston': 'male',
 'Romeo': 'male',
 'Ennis': 'male',
 'Henrietta': 'female',
 'Benito': 'male',
 'Almer': 'male',
 'Otho': 'male',
 'America': 'female',
 'Janette': 'female',
 'Lorene': 'female',
 'Otis': 'male',
 'Stuart': 'male',
 'Concepcion': 'female',
 'Neil': 'male',
 'Ophelia': 'female',
 'Parley': 'male',
 'Doss': 'male',
 'Emma': 'female',
 'Isiah': 'male',
 'Leslie': 'female',
 'Eligah': 'male',
 'Clarke': 'male',
 'Delila': 'female',
 'Elijah': 'male',
 'Norman': 'male',
 'Hope': 'female',
 'Wade': 'male',
 'Elvin': 'male',
 'Stanford': 'male',
 'Dana': 'female',
 'Verna': 'female',
 'Lucinda': 'female',
 'Liza': 'female',
 'Rosia': 'female',
 'Vallie': 'female',
 'Letha': 'female',
 'Classie': 'female',
 'Leda': 'female',
 'Parker': 'male',
 'Israel': 'male',
 'Patrick': 'male',
 'Etter': 'female',
 'Homer': 'male',
 'Steve': 'male',
 'Sophie': 'female',
 'Phebe': 'female',
 'Lemuel': 'male',
 'Rutherford': 'male',
 'Edmond': 'male',
 'Ivey': 'female',
 'Leana': 'female',
 'Christ': 'male',
 'Herschel': 'male',
 'Bertie': 'male',
 'Tyler': 'male',
 'Creola': 'female',
 'Humphrey': 'male',
 'Gerald': 'male',
 'Johanna': 'female',
 'Ashley': 'female',
 'Douglas': 'male'}

# Get the genders of names in the demographics dataset
import pandas as pd
df = pd.read_csv("../../../data/demographic_updated.csv")

def get_gender_demographics(name):
    gender_df = df[df['firstname'] == name]
    return gender_df['gender'].values[0]

# We evaulate clutrr dataset questions by giving 4 choices for the model to choose from
import random
def construct_answer_choices(original_answer):
  answer_set = ['son', 'daughter', 'father', 'mother', 'husband', 'wife', 'brother', 'sister', 'grandson', 'granddaughter', 'grandfather', 'grandmother', 'son-in-law', 'daughter-in-law', 'father-in-law', 'mother-in-law', 'brother-in-law', 'sister-in-law', 'nephew', 'neice', 'uncle', 'aunt']
  answer_set.remove(original_answer)
  random.seed(original_answer)
  final_choices = random.sample(answer_set, 3)
  final_choices.append(original_answer)
  random.shuffle(final_choices)
  return final_choices

import os
os.system("pip3 install transformers==4.20.1")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

# Generates demonstrations for the in-context learning. The length of the final total demonstrations length will not be greater than allowed_demonstrations_length.
def generate_demonstrations(allowed_demonstrations_length, new_entity, replace_demonstrations_entity=False):
    demonstrations = ""
    import random
    # allowed_demonstrations_length = 350
    random.seed(24)
    while True:
        i = random.randrange(len(train_data))
        text_story = train_data[i]['text_story'].strip()
        text_story = text_story.replace("[", "")
        text_story = text_story.replace("]", "")

        query_0 = str(train_data[i]['query'][0])
        query_1 = str(train_data[i]['query'][1])

        first_entity = train_data[i]['name_map'][query_1]
        second_entity = train_data[i]['name_map'][query_0]

        if replace_demonstrations_entity == True and first_entity!=second_entity:
            first_entity_gender = train_data_gender[first_entity]
            new_entity_gender = get_gender_demographics(new_entity)
            if first_entity_gender==new_entity_gender:
                text_story = text_story.replace(train_data[i]['name_map'][query_1], new_entity)
                first_entity = new_entity

        demonstrations+=f"context: {text_story}\n"
        demonstrations+=f"question: How is {first_entity} related to {train_data[i]['name_map'][query_0]}?\n"
        final_choices = construct_answer_choices(train_data[i]['target_gender'])
        demonstrations+=f"answer: Among the choices {final_choices[0]}, {final_choices[1]}, {final_choices[2]} and {final_choices[3]} the answer is {train_data[i]['target_gender']}.\n"
        # demonstrations+=f"answer: {train_data[i]['target_gender']}\n"
        demonstrations+="###\n"
        tokenized_prompt = tokenizer(demonstrations, return_tensors='pt')
        if len(tokenized_prompt['input_ids'][0]) > allowed_demonstrations_length:
           break
    return demonstrations

# Generates validation dataset for the in-context learning. The length of each input length will not be greater than allowed_total_length.
def generate_validation_dataset(filename, allowed_demonstrations_length, allowed_total_length, new_entity="None", replace_demonstrations_entity=False, replace_query_entity=False, baseline=True):
    # allowed_total_length = 512
    correct_ans = 0
    demonstrations = generate_demonstrations(allowed_demonstrations_length=allowed_demonstrations_length, new_entity=new_entity, replace_demonstrations_entity=replace_demonstrations_entity)
    for i in range(len(valid_data)):
        temp_prompt = demonstrations
        text_story = valid_data[i]['text_story'].strip()
        text_story = text_story.replace("[", "")
        text_story = text_story.replace("]", "")

        query_0 = str(valid_data[i]['query'][0])
        query_1 = str(valid_data[i]['query'][1])

        first_entity = valid_data[i]['name_map'][query_1]
        second_entity = valid_data[i]['name_map'][query_0]

        if replace_query_entity == True and first_entity!=second_entity:
            first_entity_gender = valid_data_gender[first_entity]
            new_entity_gender = get_gender_demographics(new_entity)
            if first_entity_gender==new_entity_gender:
                text_story = text_story.replace(valid_data[i]['name_map'][query_1], new_entity)
                first_entity = new_entity

        temp_prompt+=f"context: {text_story}\n"

        temp_prompt+=f"question: How is {first_entity} related to {valid_data[i]['name_map'][query_0]}?\n"
        final_choices = construct_answer_choices(valid_data[i]['target_gender'])
        temp_prompt+=f"answer: Among the choices {final_choices[0]}, {final_choices[1]}, {final_choices[2]} and {final_choices[3]} the answer is"

        if baseline == True:
            validation_dataset_entry = {'prompt': temp_prompt, 'label': valid_data[i]['target_gender']}
        else:
            validation_dataset_entry = {'prompt': temp_prompt, 'label': valid_data[i]['target_gender'], 'entity': new_entity}

        with open(filename,"a") as file:
            jout = json.dumps(validation_dataset_entry) + '\n'
            file.write(jout)

# Selected names for performing the entities replacments
selected_names = ['Millie', 'Brigida', 'Alina', 'Rosalia', 'Ethel', 'Elaine',
       'Desiree', 'Carrie', 'Roberta', 'Bozena', 'Nila', 'Rosalba',
       'Vincenza', 'Louella', 'Bernadette', 'Marijo', 'Juliann', 'Althea',
       'Allyson', 'Bettie', 'Della', 'Dorcas', 'Leonor', 'Marna',
       'Alisha', 'Danita', 'Raisa', 'Monica', 'Julee', 'Delynn',
       'Kathleen', 'Eva', 'Sonya', 'Trinh', 'Adriane', 'Ruthann',
       'Soledad', 'Cherie', 'Paula', 'Julianna', 'Helga', 'Charlene',
       'Deidre', 'Sallie', 'Hedy', 'Betsy', 'Janie', 'Bhavna', 'Ioana',
       'Tawnya', 'Hina', 'Marta', 'Enedina', 'Sona', 'Angela', 'Renuka',
       'Keiko', 'Loreen', 'Cari', 'Janette', 'Aviva', 'Lucretia', 'Flora',
       'Suzanne', 'Meera', 'Verna', 'Tonja', 'Irma', 'Marcia', 'Cyndi',
       'Petra', 'Armida', 'Angeline', 'Nataliya', 'Dina', 'Seema',
       'Felicia', 'Azucena', 'Thelma', 'Trena', 'Bonita', 'Katina',
       'Migdalia', 'Tabatha', 'Ivana', 'Cheryle', 'Jacqueline', 'Indira',
       'Debby', 'Neva', 'Erika', 'Penelope', 'Norma', 'Mollie', 'Ashlee',
       'Maryellen', 'Michelle', 'Chrystal', 'Lona', 'Renu']

ALLOWED_DEMONSTRATIONS_LENGTH = 350
ALLOWED_TOTAL_LENGTH = 512
# Generates the baseline dataset without entity replacement
print("Generating datasets 1/4")
generate_validation_dataset(filename="../../../data/clutrr/pre-processed/train/cluttr_val_baseline.jsonl", allowed_demonstrations_length=ALLOWED_DEMONSTRATIONS_LENGTH, allowed_total_length=ALLOWED_TOTAL_LENGTH, baseline=True)

# Generates the dataset with entity replacement only in demonstrations
print("Generating datasets 2/4")
for entity in selected_names:
    generate_validation_dataset(filename="../../../data/clutrr/pre-processed/train/cluttr_val_aug_demonstrations.jsonl", allowed_demonstrations_length=ALLOWED_DEMONSTRATIONS_LENGTH, allowed_total_length=ALLOWED_TOTAL_LENGTH, new_entity=entity, replace_demonstrations_entity=True, baseline=False)

# Generates the dataset with entity replacement only in queries
print("Generating datasets 3/4")
for entity in selected_names:
    generate_validation_dataset(filename="../../../data/clutrr/pre-processed/train/cluttr_val_aug_query.jsonl", allowed_demonstrations_length=ALLOWED_DEMONSTRATIONS_LENGTH, allowed_total_length=ALLOWED_TOTAL_LENGTH, new_entity=entity, replace_query_entity=True, baseline=False)

# Generates the dataset with entity replacement in both demonstrations and queries
print("Generating datasets 4/4")
for entity in selected_names:
    generate_validation_dataset(filename="../../../data/clutrr/pre-processed/train/cluttr_val_aug_query_and_demonstrations.jsonl", allowed_demonstrations_length=ALLOWED_DEMONSTRATIONS_LENGTH, allowed_total_length=ALLOWED_TOTAL_LENGTH, new_entity=entity, replace_demonstrations_entity=True, replace_query_entity=True, baseline=False)

print("Generated pre-processed data!")