# Top-K Recommendation with Multimodal Information

## Data Preparation
### 1) Amazon office
Amazon Review(Office) Dataset can be downloaded here(5-core).</br>
[Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)

데이터는 아래와 같이 준비되어야 합니다.</br>
```
data / ratings.csv
     / image                    / office / B01HEFLV4M.png
     / text_feature_vec.pickle
     / item_meta.json
```

- ratings.csv : Rating data. 각 열은 다음과 같습니다. userid::itemid::rating::timestamp.
- image/ : Item의 이미지가 저장된 디렉토리입니다. 이미지 파일 명은 ```item_id.png```로 저장합니다.
- text_feature_vec.pickle : Item description을 이용해서 Doc2Vec 모델을 학습한 후 추출한 text feature vector입니다. 
- item_meta.json : Item meta 정보가 담긴 dictionary 파일입니다. Item의 이미지 경로가 반드시 들어가야 하며, 예시는 아래를 참고하시기 바랍니다.

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
- Meta data, image, item description이 없는 item 제외
- 5개 미만의 item을 구매한 user 제외
- 중복 제거 </br>

(rating : 418,400, item : 18,316, user : 54,084)

### 2) Movielens-1M
Movielens Dataset can be downloaded here<br>
[Movielens 1M dataset](https://grouplens.org/datasets/movielens/1m/)

[OMDb API](http://www.omdbapi.com/)에서 크롤링을 이용하여 multimodal data 구성하였습니다.

데이터는 아래와 같이 준비되어야 합니다.</br>
```
data / ratings.csv
     / image_movielens         / poster / 1.png
     / text_feature_vec.pickle
     / user_meta.json
     / item_meta.json
```

- ratings.csv : Rating data. 각 열은 다음과 같습니다. userid::itemid::rating::timestamp.
- image_movielens/ : Item의 포스터가 저장된 디렉토리입니다. 이미지 파일 명은 ```item_id.png```로 저장합니다.
- text_feature_vec.pickle : Item의 줄거리+제목을 이용해서 Doc2Vec 모델을 학습한 후 추출한 text feature vector입니다.
- user_meta.json :  User meta 정보가 담긴 dictionary 파일입니다. 예시는 아래를 참고하시기 바랍니다.
- item_meta.json : Item meta 정보가 담긴 dictionary 파일입니다. Item의 이미지 경로가 반드시 들어가야 하며, 예시는 아래를 참고하시기 바랍니다.

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

- OMDb API에서 검색되지 않는 item 제외
- Image, title, plot(줄거리)가 없는 item 제외
- 모든 user가 평가하지 않은 item 제외

(rating : 991,276, item : 3,659, user : 6,040)

### 3)Data split

아래 코드를 이용해서 2 가지 evaluation protocol에 맞게 train-test 데이터를 나눕니다.

```
python data_split.py --data_path <Your data path/ratings.csv> --save_path <Your save path>
```
The following results will be saved in ```<Your save path>```
```
 leave-one-out/train_positive.ftr
              /test_positive.ftr
              /train_negative.ftr
              /test_negative.ftr
 ratio-split  /train_positive.ftr
              /test_positive.ftr
              /train_negative.ftr
              /test_negative.ftr
 index-info   /user_index.csv
              /item_index.csv
```

최종적으로 데이터는 아래와 같이 준비됩니다.

```
data / ratings.csv
     / image_dir /
     / text_feature_vec.pickle
     / user_meta.json(if available)
     / item_meta.json
     / leave-one-out/train_positive.ftr
                    /test_positive.ftr
                    /train_negative.ftr
                    /test_negative.ftr
     / ratio-split  /train_positive.ftr
                    /test_positive.ftr
                    /train_negative.ftr
                    /test_negative.ftr
     / index-info   /user_index.csv
                    /item_index.csv
     
```
<hr>

## Usage
