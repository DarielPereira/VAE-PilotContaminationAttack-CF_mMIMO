"""
This code is intended to simulate a pilot contamination attack in a cell-free massive MIMO network
and evaluate the detection performance using a cVAE.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Barra de progreso

# Importar funciones del proyecto
from functionsSetup import generateSetup
from functionsAllocation import PilotAssignment, AP_Assignment
from functionsChannelEstimates import channelEstimates
from functionsAttack import generateAttack, complex_to_real_batch
from functionscVAE import VAEModel
from functionsUtils import drawingSetup
import math

# --- CONFIGURACIÓN ---
configuration = {
    'nbrOfSetups': 5,  # Aumentamos número de setups para tener buenas estadísticas
    'nbrOfRealizations': 10,  # Realizaciones de canal por setup
    'L': 9,  # Número de APs
    'N': 2,  # Antenas por AP
    'K': 4,  # Número de UEs
    'T': 4,  # APs por CPU
    'tau_c': 200,  # Longitud bloque coherencia
    'tau_p': 4,  # Longitud pilotos
    'p': 100,  # Potencia UE (mW)
    'p_attacker': 5000,  # Potencia Atacante (mW)
    'cell_side': 100,  # Lado de la celda (m)
    'ASD_varphi': math.radians(10),
    'Testing': False,  # Aleatoriedad activada
    'beta_kl': 0.5,  # Peso del término KL en el score total
    'model_path': './Models/cVAE_model_NbrSamples_112500_nonNormalized.pth'  # Ruta del modelo subido
}


def run_simulation():
    # Extraer configuración
    conf = configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Ejecutando simulación en: {device}")

    # 1. Cargar el Modelo VAE Pre-entrenado
    # Calculamos input_dim basado en N: matriz de covarianza (2*N)x(2*N) aplanada
    input_dim = (2 * conf['N']) ** 2
    model = VAEModel(input_dim=input_dim, latent_dim=4, hidden_dims=[16, 8])

    if os.path.exists(conf['model_path']):
        try:
            model.load_model(conf['model_path'], map_location=device)
        except Exception as e:
            print(f"Error cargando modelo específico: {e}. Intentando carga genérica...")
            # Fallback por si la estructura de diccionarios difiere ligeramente
            model.load_state_dict(torch.load(conf['model_path'], map_location=device))
    else:
        print(f"Advertencia: No se encontró {conf['model_path']}. Asegúrate de subir el archivo.")
        return

    model.to(device)
    model.eval()

    # Contenedores para resultados
    all_scores_total = []
    all_scores_recon = []
    all_scores_kl = []
    all_labels = []  # 0: Limpio, 1: Atacado

    print(f"Iniciando evaluación sobre {conf['nbrOfSetups']} escenarios...")

    for i in tqdm(range(conf['nbrOfSetups'])):
        # 2. Generar Setup (Ubicaciones AP y UE)
        gainOverNoisedB, _, R, APpos, UEpos, _ = generateSetup(
            conf['L'], conf['K'], conf['N'], conf['T'],
            conf['cell_side'], conf['ASD_varphi'], bool_testing=conf['Testing']
        )

        # 3. Asignación de Pilotos y APs
        pilotIndex = PilotAssignment(gainOverNoisedB, conf['tau_p'], conf['K'], mode='DCC')
        # D = AP_Assignment(...) # No estrictamente necesario para la detección, solo para SE

        # 4. Generar Ataque
        # Usamos modo 'single' o 'random' para que ataque pilotos específicos
        dict_attack = generateAttack(
            conf['L'], conf['N'], conf['tau_p'], conf['cell_side'],
            conf['ASD_varphi'], conf['p_attacker'], APpos,
            bool_testing=False, attack_mode='single'
        )
        attacked_pilots = dict_attack['pilot_indices']

        # 5. Estimación de Canal (Genera B_emp)
        # B_emp shape: (N, N, L, K)
        _, _, _, _, B_emp = channelEstimates(
            R, conf['nbrOfRealizations'], conf['L'], conf['K'],
            conf['N'], conf['tau_p'], pilotIndex, conf['p'], dict_attack
        )

        # 6. Procesamiento con VAE
        # Aplanamos los datos para pasarlos por la red: (L*K, Input_Dim)
        data_tensor = complex_to_real_batch(B_emp).to(device)

        with torch.no_grad():
            x_recon, mu, logvar = model(data_tensor)

            # Calcular componentes del error por muestra
            # Error de Reconstrucción (MSE sumado sobre dimensiones)
            recon_err = ((data_tensor - x_recon) ** 2).sum(dim=1).cpu().numpy()

            # Divergencia KL (Analítica para Gaussianas)
            # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).cpu().numpy()

            # Score Total
            total_score = recon_err + conf['beta_kl'] * kl_div

        # 7. Etiquetado (Ground Truth)
        # Determinamos qué muestras corresponden a un enlace atacado
        # La función complex_to_real_batch aplana en orden: bucle L (APs) dentro de bucle K (UEs) -> NO
        # REVISAR functionsAttack.py:
        # "for l in range(L): for k in range(K): ... B_real_list.append(...)"
        # Entonces el orden es: (AP0,UE0), (AP0,UE1)... (AP1,UE0)...

        current_labels = []
        for l in range(conf['L']):
            for k in range(conf['K']):
                # Un enlace está "atacado" si el piloto del usuario k está siendo contaminado
                if pilotIndex[k] in attacked_pilots:
                    current_labels.append(1)  # Atacado
                else:
                    current_labels.append(0)  # Limpio

        # 8. Guardar resultados del setup actual
        all_scores_total.extend(total_score)
        all_scores_recon.extend(recon_err)
        all_scores_kl.extend(kl_div)
        all_labels.extend(current_labels)

    # --- Convertir a numpy para facilitar indexado ---
    all_scores_total = np.array(all_scores_total)
    all_scores_recon = np.array(all_scores_recon)
    all_scores_kl = np.array(all_scores_kl)
    all_labels = np.array(all_labels)

    # --- VISUALIZACIÓN DE HISTOGRAMAS ---
    plot_histograms(all_scores_total, all_scores_recon, all_scores_kl, all_labels)


def plot_histograms(total, recon, kl, labels):
    """Genera 3 figuras separadas para los histogramas solicitados"""

    clean_mask = (labels == 0)
    attack_mask = (labels == 1)

    print(f"\nEstadísticas Finales:")
    print(f"Muestras Limpias: {np.sum(clean_mask)}")
    print(f"Muestras Atacadas: {np.sum(attack_mask)}")

    # Configuración de estilo
    kwargs = dict(alpha=0.6, bins=50, density=True, histtype='stepfilled')

    # 1. Histograma Score Total
    plt.figure(figsize=(10, 6))
    plt.hist(total[clean_mask], color='dodgerblue', label='Enlaces Limpios', **kwargs)
    plt.hist(total[attack_mask], color='crimson', label='Enlaces Atacados', **kwargs)
    plt.title('Distribución del Score de Anomalía Total\n(Reconstrucción + beta * KL)')
    plt.xlabel('Score Total')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hist_total_score.png')

    # 2. Histograma Error de Reconstrucción
    plt.figure(figsize=(10, 6))
    plt.hist(recon[clean_mask], color='green', label='Enlaces Limpios', **kwargs)
    plt.hist(recon[attack_mask], color='orange', label='Enlaces Atacados', **kwargs)
    plt.title('Distribución del Error de Reconstrucción (MSE)')
    plt.xlabel('Error de Reconstrucción')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hist_recon_score.png')

    # 3. Histograma Divergencia KL
    plt.figure(figsize=(10, 6))
    plt.hist(kl[clean_mask], color='purple', label='Enlaces Limpios', **kwargs)
    plt.hist(kl[attack_mask], color='brown', label='Enlaces Atacados', **kwargs)
    plt.title('Distribución de la Divergencia KL')
    plt.xlabel('Divergencia KL')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hist_kl_score.png')

    print("\nGráficos generados exitosamente: 'hist_total_score.png', 'hist_recon_score.png', 'hist_kl_score.png'")
    plt.show()


if __name__ == "__main__":
    run_simulation()