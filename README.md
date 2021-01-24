# Recommendation

## Data Description
### 1) Amazon office
Amazon Review(Office) Dataset can be downloaded here<br>
[Amazon Office dataset(raw)](https://drive.google.com/drive/u/0/folders/1NMvsUaaSW9nxtMRnGcQw-8eNY1pjvAJY)

- ratingse.csv (Rating data, userid::itemid::rating::timestamp)
- item_meta.json (Item meta data)
- image.zip (Image of items)
- text_feature_vec.pickle (Text feature vector)

(rating : 418,400, item : 18,316, user : 54,084)
#### Sample item_meta.json:
```
{'B01HEFLV4M' : {'itemid': 'B01HEFLV4M',
                 'category': ['Office Products','Office & School Supplies','Desk Accessories & Workspace Organizers','Desk Supplies Holders & Dispensers','Paper Clip Holders'],
                 'description': ['Great for use around the home, office, garage or workroom. Can be attached to fridge doors, magnetic whiteboards or other flat metal surfaces for holding paper notes, reminders, recipes etc.<br>The other side of the handle has a hole so you could if you wished to place it over a nail<br> Each clip with dimension of 50mm (L) x 35mm (H) x 50mm (W); Round magnet base diameter of 27mm (1.06 inch).<br> Box Contains<br> 4 x Magnetic Clips'],
                 'title': 'LJY Magnetic Clip Fridge Paper Clips Holder, Pack of 4',
                 'also_buy': [],
                 'brand': 'LJY',
                 'feature': ['Heavy duty clips with magnet, keeping documents securely in place and damage-free.','Made of rust-resistant nickel-plated steel, sturdy and durable.','Great for use around the home, office, garage or workroom. Can be attached to fridge doors, magnetic whiteboards or other flat metal surfaces for holding paper notes, reminders, recipes etc.','Each clip with dimension of 50mm (L) x 35mm (H) x 50mm (W); Round magnet base diameter of 27mm (1.06 inch).','Package contains 4pcs magnetic clips.'],
                 'also_view': ['B01M03KZV2', 'B06XCN71JV', 'B01CT2SHIS', 'B07DHF75VV'],
                 'price': '',
                 'image_path': './image/office/B01HEFLV4M.png'}
  'B01CT2SHIS' : { ... }
    ...}
```


### 2) Movielens-1M
Movielens Dataset can be downloaded here<br>
[Movielens dataset(raw)](https://drive.google.com/drive/folders/1iRU83v1Ut8RwsH2RAlE2cYPy2iwzsEPg)

- ratings.csv (Rating data, userid::itemid::rating::timestamp)
- user_meta.json (User meta data)
- item_meta.json (Movie meta data)
- image.zip (Image of posters, 'movieid.jpg')
- text_feature_vec.pickle (Text feature vector)

(rating : 991,276, item : 3,659, user : 6,040)
#### Sample user_meta.json :
```
{'1': {'userid': '1', 'sex': 'F', 'age': '1', 'occupation': '10', 'zip_code': '48067'}
 '2': {'userid': '2', 'sex': 'M', 'age': '56', 'occupation': '16', 'zip_code': '70072'}
    ...}
```

#### Sample item_meta.json :
```
{'1': {'movieid': '1', 
       'title': 'Toy Story', 
       'year': '1995',
       'rated': 'G',
       'released': '22-Nov-95', 
       'runtime': '81 min', 
       'genre': 'Animation, Adventure, Comedy, Family, Fantasy', 
       'director': 'John Lasseter', 
       'writer': 'John Lasseter (original story by), Pete Docter (original story by), Andrew Stanton (original story by), Joe Ranft (original story by), Joss Whedon (screenplay by), Andrew Stanton (screenplay by), Joel Cohen (screenplay by), Alec Sokolow (screenplay by)', 
       'actors': 'Tom Hanks, Tim Allen, Don Rickles, Jim Varney', 
       'plot': 'A little boy named Andy loves to be in his room, playing with his toys, especially his doll named "Woody". But, what do the toys do when Andy is not with them, they come to life. Woody believes that he has life (as a toy) good. However, he must worry about Andy\'s family moving, and what Woody does not know is about Andy\'s birthday party. Woody does not realize that Andy\'s mother gave him an action figure known as Buzz Lightyear, who does not believe that he is a toy, and quickly becomes Andy\'s new favorite toy. Woody, who is now consumed with jealousy, tries to get rid of Buzz. Then, both Woody and Buzz are now lost. They must find a way to get back to Andy before he moves without them, but they will have to pass through a ruthless toy killer, Sid Phillips.', 
       'language': 'English', 
       'country': 'USA', 
       'awards': 'Nominated for 3 Oscars. Another 27 wins & 20 nominations.', 
       'rate_tomatoes': '100%', 
       'rate_metacritic': '95.0',
       'rate_imdb': '8.3', 
       'imdbvote': '868,378',
       'imdbid': 'tt0114709', 
       'type': 'movie', 
       'dvd': 'nan', 
       'boxoffice': 'nan', 
       'production': 'Pixar Animation Studios, Walt Disney Pictures', 
       'website': 'nan', 
       'response': 'True', 
       'image_path': './image_movielens/poster/1.jpg'}
  '2': { ... }
    ...}
```
  
## Data split
### Usage
```
python data_split.py --data_path <Your data path/ratings.csv> --save_path <Your save path>
```
The following results will be saved in ```<Your save path>```
```
 * leave-one-out/train_positive.ftr
                /test_positive.ftr
                /train_negative.ftr
                /test_negative.ftr
 * ratio-split/train_positive.ftr
              /test_positive.ftr
              /train_negative.ftr
              /test_negative.ftr
 * index-info/user_index.csv
             /item_index.csv
```
