# -*- coding: utf-8 -*-  
'''

 Dr Wulfrano Arturo Luna Ramírez
	wluna[at]cua.uam.mx
 Ejemplo de un agente aprendiz (usando el algoritmo Naive Bayes)
'''

import sys
from os import system, name 
import traceback
import naiveBayesModelo  as tenis


def capturaTenisUsuario():
    print(" :) #  Introduce los datos del día de hoy para saber si vamos a jugar tenis ")
    print(" :) #  Usa el siguiente formato <<obligatorio>>")
    print(" :) #    Pronóstico Climático: Soleado 1, Despejado 2, Lluvioso 3")
    print(" :) #    Temperatura: Cálido 1, Templado 2, Frío 3")
    print(" :) #    Humedad: Alta 1, Normal 2")
    print(" :) #    Viento: Fuerte 1, Débil 2")
    print(" :) #  Ejemplo 1: para un día caraterizado como:")
    print(" :) #    Pronóstico Climático: Soleado")
    print(" :) #    Temperatura: Cálido")
    print(" :) #    Humedad: Alta")
    print(" :) #    Viento: Fuerte")
    print(" :) #  Se considera una entrada de datos como sigue:")
    print(" :) #  Con valores nominales:Soleado, Caliente, Alta, Fuerte")
    print(" :) #  Con valores numéricos: 1, 1, 1, 1")
    print(" :) #  Ahora dame tus datos respondiendo a cada pregunta")
    pronostico  = input("Pronóstico climático?: ")
    temperatura = input("Temperatura?: ")
    humedad     = input("Humedad: ")
    viento      = input("Viento?: ")
    return pronostico,temperatura,humedad,viento

# Hace la prediccion con Naive Bayes en los conjuntos de datos Iris o Jugar Tenis
#
# Codigo de valores de Tennis
# Sunny 1, Overcast 2, Rain 3
# Hot 1, Mild 2, Cool 3
# High 1, Normal 2
# Strong 1, Weak 2
#

    return (origen,destino)

# Funcion principal
def main():
    try: 
        print(" ####################################################################")
        print(" #                                                                  #")
        print(" # Hola soy el agente aprendiz LEANDRO  :-)                         #")
        print(" #  Se decidir cuándo ir a jugar tenis.                             #")
        print(" # Mi función de decisión fue desarrollada con el algoritmo         #")
        print(" # Naive Bayes, un método sencillo pero efectivo                    #")
        print(" # para la clasificación y la toma de decisiones.                   #")
        print("#####################################################################")
        print(" # Naive Bayes es un algoritmo de clasificación probabilística.     #")
        print(" # Se basa en el Teorema de Bayes:                                  #")
        print(" #               P(C|X) = P(X|C)P(C)/P(X)                           #")
        print(" # en la suposición de independencia entre las variables para       #")
        print(" # calcular la probabilidad de cada clase.                          #")
        print(" #                                                                  #")
        print("#####################################################################")

        print("¿Desea cargar el entrenamiento o volver a entrenar?")
        print("1. Cargar         2. Reentrenar")
        opcion = input("Opcion : ")    
    
        modelo = None

        if opcion == "1":
            print(" :) #  Cargando el modelo...")
            modelo = tenis.carga_modelo("modelo_tenis.mod")
            if modelo is None:
                print(" :) #  No se encontró el modelo. Entrenando uno nuevo...")
                datos = tenis.cargaTenis()
                modelo = tenis.entrena_y_guarda(datos, "modelo_tenis.mod")
        else:
            print(" :) #  Entrenando nuevo modelo y guardando...")
            datos = tenis.cargaTenis()
            modelo = tenis.entrena_y_guarda(datos, "modelo_tenis.mod")
        
        p,t,h,v = capturaTenisUsuario()
        renglon = [int(p),int(t),int(h),int(v)]
        
        print(" :) #  Haciendo predicción")
        tenis.hazPrediccionConModelo(renglon, modelo)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()


