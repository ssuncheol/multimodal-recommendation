# Recommendation

## Data Description
### 1) Amazon office
Amazon Review(Office) Dataset can be downloaded here<br>
[Amazon Office dataset(raw)](https://drive.google.com/drive/u/0/folders/1NMvsUaaSW9nxtMRnGcQw-8eNY1pjvAJY)

- ratings_Office_Products_5.csv (Rating data, userid::itemid::rating::timestamp)
- office_image.zip (Image of items)
- reviews_Office_Products_5.json.gz (Subset of the review data in which all users and item have at least 5 reviews)

### Sample review:
<pre>
<code> {
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}</code></pre>
where

### 2) Movielens-1M
Movielens Dataset can be downloaded here<br>
[Movielens dataset(raw)](https://drive.google.com/drive/folders/1iRU83v1Ut8RwsH2RAlE2cYPy2iwzsEPg)

- ratings.csv (Rating data, userid::itemid::rating::timestamp)
- user_movielens.json (User meta data)
- item_movielens.json (Movie meta data)
- image_movielens.zip (Image of posters, 'movieid.jpg')

### Sample user_movielens.json :
<pre>
<code> {
  '1': {'userid': '1', 'sex': 'F', 'age': '1', 'occupation': '10', 'zip_code': '48067'}
  '2': {'userid': '2', 'sex': 'M', 'age': '56', 'occupation': '16', 'zip_code': '70072'}
    ...
  }</code></pre>

### Sample item_movielens.json :
<pre>
<code> 
{
  '1': {'movieid': '1', 
        'title': 'Toy Story', 
        'year': '1995',
        'rated': 'G',
        'released': '22-Nov-95', 
        'runtime': '81 min', 'genre': 'Animation, Adventure, Comedy, Family, Fantasy', 'director': 'John Lasseter', 
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
        'type': 'movie', 'dvd': 'nan', 'boxoffice': 'nan', 
        'production': 'Pixar Animation Studios, Walt Disney Pictures', 
        'website': 'nan', 
        'response': 'True', 
        'image_path': './image_movielens/poster/1.jpg'}
  '2': { ... }
    ...
  }</code></pre>
