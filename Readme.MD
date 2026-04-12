# Système de Surveillance Intelligente par Caméra
### Filière : Intelligence Artificielle et Cybersécurité
**École Nationale des Sciences Appliquées de Béni Mellal — USMS**  
Année Universitaire 2025–2026

---

## Auteurs
- **ALIOUALI Othmane**
- **ELJARIDA Saad Eddine**

Encadré par : **Pr. ABOUHILAL Abdelmoula**

---

## Description

Système de surveillance intelligente basé sur une architecture **Edge-to-Cloud** :
- **Edge** : Raspberry Pi 5 + Hailo-8 AI HAT+ (26 TOPS) — détection temps réel via YOLO11n
- **Cloud** : Supabase (PostgreSQL) + Dashboard Next.js 14 — supervision et alertes

Le système détecte automatiquement les comportements anormaux :
- Intrusion dans une zone interdite
- Chute d'une personne
- Objet abandonné
- Attroupement

---

## Architecture

```
Webcam / Caméra CSI
        │
        ▼
Raspberry Pi 5 + Hailo-8
  ├── libcamera (capture)
  ├── YOLO11n.hef (inférence NPU 26 TOPS)
  ├── NMS + BoT-SORT (tracking)
  └── FastAPI (API REST + WebSocket)
        │
        ▼ HTTPS / JSON
        │
   Supabase (PostgreSQL)
        │
        ▼ Realtime
        │
   Dashboard Next.js 14 (Vercel)
```

---

## Stack technique

| Couche | Technologie | Rôle |
|---|---|---|
| Inférence IA | YOLO11n + Hailo-8 | Détection objets temps réel |
| Tracking | BoT-SORT | Suivi multi-objets |
| API Backend | FastAPI + Python | REST + WebSocket |
| Base de données | Supabase (PostgreSQL) | Persistance alertes |
| Dashboard | Next.js 14 + shadcn/ui | Interface supervision |
| ORM / Graphiques | Prisma + Recharts | Accès BDD + visualisation |

---

## Structure du projet

```
Phase-4/
├── main.py              # API FastAPI — routes REST + WebSocket
├── models.py            # Modèles Pydantic — validation données
├── services.py          # Analyse comportementale — règles de détection
├── database.py          # Client Supabase + broadcast WebSocket
├── simulator.py         # Simulation pipeline RPi5 sur PC (webcam + ONNX)
├── convert_to_onnx.py   # Conversion YOLO11n PT → ONNX
├── test_inference.py    # Test inférence ONNX webcam temps réel
├── test_pt_vs_onnx.py   # Comparaison PT vs ONNX
├── requirements.txt     # Dépendances Python
├── .gitignore           # Fichiers exclus du dépôt
└── README.md            # Ce fichier
```

---

## Installation

### Prérequis
- Python 3.11+
- Node.js 20+
- Compte Supabase (gratuit)

### 1. Cloner le dépôt
```bash
git clone https://github.com/othmane1244/Phase-4.git
cd Phase-4
```

### 2. Créer l'environnement virtuel
```bash
python -m venv hailo_env
# Windows
hailo_env\Scripts\activate
# Linux / RPi 5
source hailo_env/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement
Créer un fichier `.env` à la racine :
```
SUPABASE_URL=https://votre-projet.supabase.co
SUPABASE_SERVICE_KEY=votre-clé-service-role
API_HOST=127.0.0.1
API_PORT=8000
CAMERA_ID=cam_01
```

### 5. Convertir le modèle YOLO11n
```bash
python convert_to_onnx.py
```
Cela télécharge `yolo11n.pt` et génère `yolo11n.onnx`.

---

## Lancement (simulation sur PC)

**Terminal 1 — API FastAPI :**
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Simulateur pipeline :**
```bash
python simulator.py
```

**Interface Swagger :** `http://127.0.0.1:8000/docs`  
**Alertes en temps réel :** `http://127.0.0.1:8000/alerts/buffer/`

---

## Dashboard Next.js 14

### Structure des fichiers

```
dashboard/
├── app/
│   ├── layout.tsx          — Layout général + métadonnées
│   ├── page.tsx            — Page principale du dashboard (SSR)
│   └── globals.css         — Styles globaux Tailwind
├── components/
│   ├── AlertList.tsx       — Liste alertes temps réel (Supabase Realtime)
│   ├── StatsCards.tsx      — Cartes statistiques
│   └── AlertBadge.tsx      — Badge coloré par type d'alerte
├── lib/
│   └── supabase.ts         — Client Supabase + types TypeScript
├── .env.local              — Clés Supabase (ne pas committer)
└── package.json
```

### Installation

```bash
cd dashboard
npm install
npm install @supabase/supabase-js lucide-react recharts
```

### Variables d'environnement `.env.local`

```
NEXT_PUBLIC_SUPABASE_URL=https://votre-projet.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=votre-clé-anon-publique
```

> ⚠️ Utiliser la clé **`anon`** (publique) pour le dashboard — jamais `service_role`.

### Lancement

```bash
npm run dev
```

Ouvre **`http://localhost:3000`**

### Fonctionnalités

| Composant | Description |
|---|---|
| `StatsCards` | 6 cartes : total, non résolues, intrusions, chutes, objets abandonnés, attroupements |
| `AlertList` | Liste temps réel via Supabase Realtime — sans rechargement |
| `AlertBadge` | Badge coloré : 🚨 Intrusion · 🆘 Chute · 🎒 Objet abandonné · 👥 Attroupement |
| Pipeline panel | Latences étape par étape Edge-to-Cloud |

### Lancement complet (3 terminaux)

**Terminal 1 — API FastAPI :**
```bash
cd "T3 dachbord" && hailo_env\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Dashboard :**
```bash
cd dashboard && npm run dev
```

**Terminal 3 — Simulateur webcam :**
```bash
cd "T3 dachbord" && hailo_env\Scripts\activate
python simulator.py
```

---

## Règles de détection

| Comportement | Condition | Seuil |
|---|---|---|
| Intrusion | Centre bbox personne dans `ZONE_INTERDITE` | Zone : 40% gauche |
| Chute | Ratio w/h bbox personne > seuil | `FALL_RATIO_THRESHOLD = 1.2` |
| Objet abandonné | Bagage sans personne à proximité | `ABANDONED_OBJECT_DIST = 120px` |
| Attroupement | Nb personnes dans le frame ≥ seuil | `CROWD_MIN_PERSONS = 3` |

Tous les seuils sont configurables dans `services.py`.

---

## Performances (estimations)

| Plateforme | Modèle | FPS estimé |
|---|---|---|
| PC (i7 + RTX) | ONNX (dev) | ~25–35 FPS |
| RPi 5 seul | ONNX | ~6–8 FPS |
| RPi 5 + Hailo-8 | HEF (INT8) | ~25–35 FPS |

---

## Phases du projet

- [x] **Phase 1** — Analyse et sélection plateforme matérielle
- [x] **Phase 2** — Sélection modèle IA (YOLO11n)
- [x] **Phase 3** — Architecture logicielle Edge-to-Cloud
- [x] **Phase 4** — Conversion modèle + API FastAPI + Supabase
- [x] **Phase 5** — Dashboard Next.js 14
- [ ] **Phase 6** — Déploiement RPi 5 + Hailo-8 (en attente matériel)
- [ ] **Phase 7** — Tests et benchmarks réels

---

## Notes importantes

> Les valeurs de latence et de FPS présentées sont des **estimations** basées sur
> la documentation officielle Hailo Model Zoo. Des benchmarks réels seront
> effectués lors de la phase de déploiement sur RPi 5 + Hailo-8.

> Le fichier `.env` contenant les clés Supabase n'est **jamais** publié sur GitHub.
> Les modèles `.pt` et `.onnx` sont exclus du dépôt (taille > 5 MB).

---

## Licence

Projet académique — ENSA Béni Mellal 2025–2026. Tous droits réservés.