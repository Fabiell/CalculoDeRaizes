import numpy as np
import matplotlib.pyplot as plt

def derivada_avancada(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def derivada_recuada(f, x, h=1e-5):
    return (f(x) - f(x - h)) / h

def derivada_centrada(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def newton_raphson(f, x0, metodo_derivada, tol=1e-10, max_iter=100):
    """
    Implementa o método de Newton-Raphson para encontrar raízes.
    
    Parâmetros:
    - f: função para encontrar a raiz
    - x0: chute inicial
    - metodo_derivada: função para calcular a derivada
    - tol: tolerância para convergência
    - max_iter: número máximo de iterações
    
    Retorna: raiz encontrada, número de iterações
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i + 1

        fpx = metodo_derivada(f, x) 
        
        if abs(fpx) < 1e-12:
            print(f"Derivada muito pequena em x = {x}. O método falhou.")
            return None, i + 1
        
        x_novo = x - fx / fpx

        if abs(x_novo - x) < tol:
            return x_novo, i + 1
        
        x = x_novo
    
    print(f"Não convergiu após {max_iter} iterações. Último x: {x}")
    return x, max_iter

metodos = [
    ("Avançada", derivada_avancada),
    ("Recuada", derivada_recuada),
    ("Centrada", derivada_centrada)
]

print("\n" + "=" * 70)
print("DESAFIO 1: 2^x = x² → f(x) = 2^x - x²")
print("Encontrando as 3 raízes com cada tipo de derivada")
print("=" * 70)

def f1(x):
    return 2**x - x**2

chutes_f1 = [-0.8, 2, 4]

for nome_metodo, metodo in metodos:
    print(f"\n{'─' * 70}")
    print(f"Derivada {nome_metodo}:")
    print(f"{'─' * 70}")
    for i, chute in enumerate(chutes_f1, 1):
        raiz, iter = newton_raphson(f1, chute, metodo)
        if raiz is not None:
            print(f"  Raiz {i}: x* = {raiz:10.8f} | f(x*) = {f1(raiz):.2e} | Iterações: {iter}")
        else:
            print(f"  Raiz {i}: Não foi possível encontrar a raiz com chute inicial {chute}.")

print("\n" + "=" * 70)
print("DESAFIO 2: tan(x) = 1/x → f(x) = tan(x) - 1/x")
print("Encontrando as 5 primeiras raízes positivas")
print("=" * 70)

def f2(x):
    return np.tan(x) - 1/x

chutes_f2 = [0.5, 4.5, 7.7, 10.9, 14.1] 

resultados_desafio2 = {}

print(f"\nComparando métodos de derivada para cada raiz:")
print(f"{'=' * 70}")

for i, chute in enumerate(chutes_f2, 1):
    print(f"\nRaiz {i} (Chute inicial: {chute}):")
    print(f"{'─' * 30}")
    resultados_desafio2[i] = {}
    
    for nome_metodo, metodo in metodos:
        raiz, iteracoes = newton_raphson(f2, chute, metodo, tol=1e-10)
        
        if raiz is not None:
            print(f"  {nome_metodo:8}: x* = {raiz:10.8f} | Iterações: {iteracoes}")
            resultados_desafio2[i][nome_metodo] = iteracoes
        else:
            print(f"  {nome_metodo:8}: Não convergiu.")
            resultados_desafio2[i][nome_metodo] = None

print(f"\n{'=' * 70}")
print("RESUMO COMPARATIVO (Iterações até convergência):")
for i in resultados_desafio2:
    res = resultados_desafio2[i]
    partes = []
    for nome, iters in res.items():
        if iters is not None:
            partes.append(f"{nome} ({iters})")
        else:
            partes.append(f"{nome} (Falhou)")
    print(f"Raiz {i}: {', '.join(partes)}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax1 = axes[0]
x1 = np.linspace(-1, 5, 400)
y1_a = 2**x1
y1_b = x1**2
ax1.plot(x1, y1_a, 'b-', linewidth=2, label='y = $2^x$')
ax1.plot(x1, y1_b, 'r-', linewidth=2, label='y = $x^2$')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Desafio 1: $2^x = x^2$')
ax1.legend()
ax1.set_ylim(-1, 20)

ax2 = axes[1]
ax2.plot(x1, f1(x1), 'g-', linewidth=2, label='$f(x) = 2^x - x^2$')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('Desafio 1: Função $f(x) = 2^x - x^2$')
ax2.legend()
ax2.set_ylim(-10, 10)

ax3 = axes[2]
x2 = np.linspace(0.1, 15, 1000)

y2_tan = np.tan(x2)
y2_inv = 1/x2

y2_tan[np.abs(y2_tan) > 10] = np.nan
ax3.plot(x2, y2_tan, 'b-', linewidth=2, label='y = tan(x)', alpha=0.7)
ax3.plot(x2, y2_inv, 'r-', linewidth=2, label='y = 1/x')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Desafio 2: tan(x) = 1/x')
ax3.legend()
ax3.set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('newton_raphson_resultados.png', dpi=150, bbox_inches='tight')

print("Gráficos salvos em 'newton_raphson_resultados.png'")
