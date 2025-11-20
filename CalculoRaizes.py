import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, x0, tol=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-15:
            print(f"Derivada muito pequena em x = {x}")
            return None, i
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, i + 1
        
        x = x_new
    
    print(f"Não convergiu após {max_iter} iterações")
    return x, max_iter

print("=" * 60)
print("DESAFIO 1: 2^x = x²")
print("Transformando em: f(x) = 2^x - x² = 0")
print("=" * 60)


def f1(x):
    return 2**x - x**2

def df1(x):
    return 2**x * np.log(2) - 2*x

pontos_iniciais_1 = [-1.0, 2.0, 4.5]
solucoes_1 = []

for x0 in pontos_iniciais_1:
    raiz, iter_count = newton_raphson(f1, df1, x0)
    if raiz is not None:
        eh_nova = True
        for sol in solucoes_1:
            if abs(raiz - sol) < 1e-6:
                eh_nova = False
                break
        if eh_nova:
            solucoes_1.append(raiz)
            print(f"\nPonto inicial x0 = {x0}:")
            print(f"  Raiz encontrada: x = {raiz:.10f}")
            print(f"  Verificação: 2^x = {2**raiz:.10f}, x² = {raiz**2:.10f}")
            print(f"  f(x) = {f1(raiz):.2e}")
            print(f"  Iterações: {iter_count}")

print(f"\nTotal de soluções encontradas: {len(solucoes_1)}")

print("\n" + "=" * 60)
print("DESAFIO 2: tan(x) = 1/2")
print("Transformando em: f(x) = tan(x) - 1/2 = 0")
print("=" * 60)

def f2(x):
    return np.tan(x) - 0.5

def df2(x):
    return 1 / (np.cos(x)**2)

solucoes_2 = []
base = np.arctan(0.5)

for k in range(5):
    x0 = base + k * np.pi
    raiz, iter_count = newton_raphson(f2, df2, x0)
    if raiz is not None and raiz > 0:
        solucoes_2.append(raiz)
        print(f"\nRaiz {k+1}:")
        print(f"  x = {raiz:.10f} rad ({raiz*180/np.pi:.6f}°)")
        print(f"  tan(x) = {np.tan(raiz):.10f}")
        print(f"  f(x) = {f2(raiz):.2e}")
        print(f"  Iterações: {iter_count}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x1 = np.linspace(-2, 5, 400)
axes[0].plot(x1, 2**x1, 'b-', linewidth=2, label='2^x')
axes[0].plot(x1, x1**2, 'r-', linewidth=2, label='x²')
for sol in solucoes_1:
    axes[0].plot(sol, 2**sol, 'go', markersize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Desafio 1: 2^x = x²')
axes[0].legend()
axes[0].set_ylim(0, 20)

x2 = np.linspace(0, 5*np.pi, 1000)
y_tan = np.tan(x2)
y_tan = np.where(np.abs(y_tan) > 10, np.nan, y_tan)
axes[1].plot(x2, y_tan, 'b-', linewidth=2, label='tan(x)')
axes[1].axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='y = 1/2')
for sol in solucoes_2:
    axes[1].plot(sol, 0.5, 'go', markersize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('x (rad)')
axes[1].set_ylabel('y')
axes[1].set_title('Desafio 2: tan(x) = 1/2')
axes[1].legend()
axes[1].set_ylim(-3, 3)

plt.tight_layout()
plt.show()