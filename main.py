#se importan las librerias
from fastapi import FastAPI, HTTPException
import pandas as pd
import calendar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#se asigna FastAPI al variable app para un mejor manejo de codigo
app=FastAPI() 


df=pd.read_csv('dataset.csv') # se lee el documento csv
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce') #se cambia el formato de fecha para poder determinar los dias meses y años necesarios en las apis

# esta parte de codigo es usada para el modelo de recomendacion usando la similitud del coseno 
df2=df  #se genera una nueva variable con el dataframe
#para el modelo de recomendacion se pretende utilizar las columnas overview y generos, esto permitira un mejor filtrado de peliculas similares
df2['overview']=df2['overview'].fillna('') # se rellenan los elementos nulos o NaN
df2['generos']=df2['generos'].fillna('')
df2['combinados']=df2['overview']+" "+df2['generos'].apply(lambda x: " ".join(eval(x)))  # se crea una columna nueva con los valores de las columnas selccionadas 
#esta union dara como resultado el texto encontrado en el overview y al final se encontraran los generos a los cuales pertenece esa pelicula

tfidf=TfidfVectorizer(stop_words='english') #se guarda el metodo de vectorizado con la condicionante de que todas las palabras o conectores comunes del idioma ingles  se ignores
tfidf_matrix=tfidf.fit_transform(df2['combinados']) #se aplica el metodo de vectorizado a la columna generada previamente
similitud_coseno=cosine_similarity(tfidf_matrix, tfidf_matrix) #se genera la matriz de similutd del coseno, entre el valor sea mas cercano a 1 mas similares son

def get_recomendacion(titulo): # funcion para obtener las 5 peliculas similares al titulo provisto
    indice=df2.index[df2['title']==titulo].tolist() # se busca el indice del titulo ingresado
    if not indice:  # si el titulo no existe no devuelve nada
        return None
    indice=indice[0]

    similares=list(enumerate(similitud_coseno[indice])) # se busca los valores similares al indice obtenido
    similares=sorted(similares, key=lambda x:x[1], reverse=True)[1:6] # se ordenan los resultados y se obtienen solo los 5 primeros (se coloca 1:6 porque el titulo provisto tambien puede salir como resultado, eso lo elimina de la lista
    peliculas=[i[0] for i in similares] # se guardan los resultados de los indicees encontrados en la variable peliculas
    pelis=df2['title'].iloc[peliculas].tolist() # se guardan los titulos de las peliculas
    return pelis


#api para obtener la cantidad de filmaciones por mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes (mes: str):
    meses=["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"] 
    # se genera la lista meses para poder determinar un indice y poder compararlo con el metodo dt.month
    try:
        numero_mes=list(meses).index(mes.lower())  #se coloca el valor en minusculas independiente de como se coloque el input (para evitar errores)
    except ValueError: # si existe error de input las excepciones permiten desplegar un mensaje evitando que la api se detenga
        raise HTTPException(status_code=400, detail="mes no valido")
    
    cantidad=df[df['release_date'].dt.month==numero_mes].shape[0] # se filtra en un df la cantidad de peliculas extrenadas en el mes provisto se entrega el resultado con el tamaño del vector
    return {"mensaje":f"{cantidad} cantidad de pelicula fueron estrendas en el mes de {mes}"}

#api para obtener la cantidad de filmaciones por dia
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia (dia: str):
    try:
        dias={'Monday':'lunes', 'Friday':'viernes', 'Thursday':'jueves', 'Wednesday':'miercoles', 'Saturday':'sabado', 'Tuesday':'martes',
       'Sunday':'domingo'} # se genera el siguiente diccionario ya que el metodo day_name genera una lista con los nombres de los dias en ingles
        df['dia_semana']=df['release_date'].dt.day_name()
        df['dia_semana']=df['dia_semana'].replace(dias) # se reemplazan los valores en ingles por valores en español en la columna generada para los dias de la semana 
        dia_peliculas=df[df['dia_semana'].str.lower()==dia.lower()]
        cantidad_peliculas=dia_peliculas.shape[0] #al igual que con los meses se obtiene la cantida de peliculas
    except ValueError:
        raise HTTPException(status_code=400, detail="dia no valido")
    
    return {"mensaje":f"{cantidad_peliculas} cantidad de peliculas fueron estrenadas los dias {dia}"}

#api para obtener el score por pelicula y
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo:str):
    try:
        filmacion=df[df['title']==titulo] # se busca el titulo dentro del dataframe
        if filmacion.empty:
            return f"La pelicula {titulo} no se encontro en la base de datos" # si no se encuentra la pelicula se despliega este mensaje
        score=filmacion['popularity'].values[0] # se obtiene el valor de popularidad y año de estreno
        anio_filme=filmacion['release_year'].values[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Titulo no valido")
    return f"La pelicula {titulo} estrenada en el año {anio_filme} con un score de {score}" 

#api para obtener los votor del filme
@app.get("/votos_titulo/{titulo}") 
def votos_titulo(titulo: str):
    try:
        filme=df[df['title']==titulo] #se busca el titulo en el dt
        if filme.empty:
            return f"La pelicula {titulo} no se encontro en la base de datos"
        votos=filme['vote_count'].values[0]
        votos_promedio=filme['vote_average'].values[0]
        anio_filme=filme['release_year'].values[0] # se obtiene la informacion (como en la api anterior)
        if votos < 2000: #si la cantidad de votos no es la suficiente  se despliega un mensaje estipulando que el filme no cumple con el criterio
            return f"La pelicula {titulo} no cumple con el minimo de votos"

    except ValueError:
        raise HTTPException(status_code=400, detail="Titulo no valido")
    
    return (f'La pelicula {titulo} fue estrenada en el año {anio_filme}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {votos_promedio}')


#api para obtener el exito del actor 
@app.get("/actores/{actor}")
def get_actor(actor: str):
    try:
        actordf=df[df['actores'].str.contains(actor, na=False)] # se busca dentro de la columna actores todas las columnas que contengan el nombre del actor
        if actordf.empty:
            return f"El actor {actor} no se encontro en la base de datos" # en caso de no encontrarlo se despliega un mensaje
        peliculas_actor=actordf.shape[0] #se obtiene un vector con la cantidad de las peliculas en las cual aparece ese actor
        retorno_peliculas=actordf['revenue'].sum() #se hace la suma de revenue generado por todas las peliculas
        promedio_peliculas=retorno_peliculas/peliculas_actor #se obtiene el promedio de revenue total entre la cantidad de peliculas en las que aparece
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Nombre de actor no valido")     #mensaje de error en caso de colocar un parametro invalido 
    return f"El actor {actor} ha participado de {peliculas_actor} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_peliculas} con un promedio de {promedio_peliculas} por filmacion"

#api para obtener el exito del director
@app.get("/directores/{director}")
def get_director(director:str):
    try:
        dfdirector=[] #se genera una lista en la cual se guardaran todas las peliculas que ha dirigido el director
        for _,row in df.iterrows():
            puestos=eval(row['puestos_produccion']) #se verifica que el nombre provisto exista
            nombres=eval(row['nombre_produccion']) #se verifica que el nombre provisto sea de direccion
        
            for puestos, nombres in zip(puestos, nombres): # se zippean los resultados donde aparece el director y se iteran los nombres y puestos
                if puestos=="Director" and nombres == director: # si se encuentra el nombre y su cargo es director se guardan los datos de cada pelicula
                    dfdirector.append({"Pelicula":row['title'],
                                       "Fecha de estreno":row['release_date'],
                                       "Costo":row['budget'],
                                       "Retorno":row['revenue'],
                                       "Ganancia":row['return']})
                    break #una vez encontrado se ha encontrado al director se termina la busqueda en la fila

    except ValueError:
        raise HTTPException(status_code=400, detail="Nombre del director no valido") 

    return f" El director {director} ha tenido el siguiente exito: {dfdirector} "

#api para obtener recomendacion de peliculas a partir de un titulo
@app.get("/recomendacion/{titulo}")
async def recomendacion(titulo: str): #se usa async para poder hacer en analisis de forma asincrna de la funcion previa y mejorar el rendimiento de la api
    recomendaciones=get_recomendacion(titulo)  #se manda a llamar la funcion previamente descrita del modelo
    if recomendaciones is None: #si no existe la pelicula se despliega el mensaje
        raise HTTPException(status_code=404, detail="Pelicula no encontrada") 
    return  f"Peliculas similares a {titulo} recomendadas: {recomendaciones}"  # resultado de titulos similares.
#C:\Users\Jesus\Documents\Analisis de Datos\Proyecto\main.py