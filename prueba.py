L = [10, 20, 30, 40, 50]

# Quitar hasta vaciar, procesando cada elemento retirado
while L:
    x = L.pop()      # quita y retorna el último
    print("quitado:", x, "lista ahora:", L)