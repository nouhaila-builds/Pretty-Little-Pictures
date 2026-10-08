export type Lang = "en" | "fr";

export type Copy = {
  langLabel: string;
  kicker: string;
  title: string;
  lede: string;
  question: string;
  tags: string[];
  measured: string;
  pending: string;
  s1Title: string;
  s1: string;
  s2Title: string;
  s2: string;
  s3Title: string;
  s3Note: string;
  sliderLabel: string;
  visible: string;
  realInfrared: string;
  noPair: string;
  s4Title: string;
  s4Intro: string;
  counts: (train: string, val: string, test: string, one: string) => string;
  countsMissing: string;
  s5Title: string;
  archLabel: string;
  archRgb: string;
  archModel: string;
  archOut: string;
  s5: string;
  s6Title: string;
  s6: string;
  fractionLabel: string;
  awaiting: string;
  sameList: string;
  noCurve: string;
  generated: string;
  panelAlt: (id: string) => string;
  chartCaption: string;
  hintCloser: string;
  hintLower: string;
  hintBetter: string;
  chartAria: (label: string, hint: string, ticks: string) => string;
  valLoss: string;
  epoch: string;
  curveLabel: string;
  s7Title: string;
  protocol: [string, string][];
  seedNote: string;
  s7Metrics: string;
  finding: (p: { min: string; max: string; monotonic: boolean; ssimFrom: string; ssimTo: string; lpipsFrom: string; lpipsTo: string; apFrom: string; apTo: string }) => string;
  findingPending: string;
  findingClose: string;
  unetRole: string;
  tableNote: string;
  colImages: string;
  unetRow: string;
  diffusionRow: (percent: number) => string;
  caveatKicker: string;
  caveat: string;
  caveatUse: string;
  s8Title: string;
  s8: string;
  s8Note: string;
  reviewMissing: string;
  scenes: Record<string, string>;
  pairAlt: (id: string) => string;
  visibleShort: string;
  rank: (model: string, worst: string, mae: string, count: number, total: number, prefix: string) => string;
  modelName: (baseline: boolean, percent: number) => string;
  s9Title: string;
  s9: string;
  s10Title: string;
  s10a: string;
  s10b: string;
  s11Title: string;
  s11: string;
  status: (value: string) => string;
};

const en: Copy = {
  langLabel: "Language",
  kicker: "Independent research case study · 2026",
  title: "Pretty little pictures",
  lede: "Looks can be deceiving — even to a machine.",
  question: "When a visible-to-infrared translator is trained on progressively fewer aligned pairs, do pixel fidelity, perceptual similarity, and pedestrian-detection utility degrade together?",
  tags: ["Generative AI", "Computer vision", "RGB to infrared", "Multimodal learning", "Data scarcity", "PyTorch"],
  measured: "Measured on the LLVIP test set",
  pending: "Experiments pending",
  s1Title: "We see in RGB",
  s1: "Most machine perception starts from an ordinary camera. The image is dense, cheap, and full of texture. That is the modality we already know how to collect.",
  s2Title: "Until RGB cannot see enough",
  s2: "Under weak light, visible texture thins out. A person can remain in the scene after the colour image has lost them. Thermal sensing is a different measurement of that same place, not a universally better photograph.",
  s3Title: "Same scene. Different sensor.",
  s3Note: "One LLVIP test frame. Drag to compare the visible image with the aligned infrared image. The infrared side is the real capture, not a generated one.",
  sliderLabel: "Divider between the visible image and the real infrared image",
  visible: "Visible",
  realInfrared: "Real infrared",
  noPair: "No aligned pair has been exported yet.",
  s4Title: "The paired set is the scarce one",
  s4Intro: "LLVIP is the study set: aligned visible and long-wave infrared pairs, mostly at night, with pedestrian boxes.",
  counts: (train, val, test, one) => `The fixed lists contain ${train} training pairs, ${val} validation pairs, and ${test} test pairs. The 1% list contains ${one} pairs.`,
  countsMissing: "Pair counts stay unstated until prepare_dataset.py reads the archive.",
  s5Title: "Can we generate what is missing?",
  archLabel: "RGB to infrared pipeline",
  archRgb: "RGB image",
  archModel: "Conditional model",
  archOut: "Synthetic infrared",
  s5: "The baseline is an L1 U-Net. The main model is a small conditional diffusion network: the RGB image is concatenated with the noisy infrared image, and the network predicts the noise. It is trained from the pairs. It is not a photographic foundation model.",
  s6Title: "Now remove the data",
  s6: "The test set stays fixed. A single random permutation is drawn with seed 42, and the smaller training lists are nested prefixes of that same permutation: 1% ⊂ 5% ⊂ 20% ⊂ 100%. The control below reads only the exported record for that fraction. Each panel is RGB, then the generated infrared, then the real infrared.",
  fractionLabel: "Training data fraction",
  awaiting: "Awaiting experiment",
  sameList: "Same fixed test list for every fraction.",
  noCurve: "Training curve not yet available",
  generated: "Generated",
  panelAlt: (id) => `RGB, generated infrared, and real infrared for ${id}`,
  chartCaption: "Diffusion only. Each panel starts at zero and has its own vertical scale. The horizontal axis is the training fraction.",
  hintCloser: "higher is closer",
  hintLower: "lower is closer",
  hintBetter: "higher is better",
  chartAria: (label, hint, ticks) => `${label}, ${hint}, across training fractions ${ticks} percent`,
  valLoss: "Validation loss",
  epoch: "Epoch",
  curveLabel: "Validation loss by epoch",
  s7Title: "Pretty is not the same as useful",
  protocol: [
    ["Dataset", "LLVIP"],
    ["Training pairs", ""],
    ["Validation pairs", ""],
    ["Test pairs", ""],
    ["Training fractions", "100%, 20%, 5%, 1%"],
    ["Split", "One random permutation, seed 42; nested prefixes"],
    ["Generative model", "Conditional diffusion"],
    ["Baseline", "L1 U-Net, 100% training data only"],
    ["Diffusion training", "30 epochs per fraction"],
    ["Evaluation", "PSNR, SSIM, LPIPS, pedestrian AP"],
    ["Detector", "Frozen pedestrian detector trained on real infrared"],
  ],
  seedNote: "One training run per fraction, seed 42. These results are an independent case study, not a multi-seed benchmark.",
  s7Metrics: "PSNR and SSIM measure agreement with the aligned infrared target. LPIPS measures perceptual distance. Pedestrian AP asks a different question: whether the generated infrared remains useful to a detector trained on real infrared.",
  finding: (p) =>
    `Across the four diffusion training fractions, the metrics did not degrade together. PSNR stayed between ${p.min} and ${p.max} dB${p.monotonic ? "." : " and was not monotonic."} SSIM decreased from ${p.ssimFrom} to ${p.ssimTo}, while LPIPS increased from ${p.lpipsFrom} to ${p.lpipsTo}. Pedestrian AP on the generated images fell from ${p.apFrom} at 100% training data to ${p.apTo} at 1%.`,
  findingPending: "The comparison stays unstated until the export contains every score.",
  findingClose: "The result is not one notion of quality. Pixel error, perceptual similarity, and downstream detection can tell different stories.",
  unetRole: "The U-Net is a supervised L1 reference trained on the full training set. Data-scarcity ablations were performed only with the diffusion model.",
  tableNote: "Same 3,463 test images. Higher PSNR, SSIM, and AP are better. Lower LPIPS is closer.",
  colImages: "Test images",
  unetRow: "U-Net reference, 100%",
  diffusionRow: (percent) => `Diffusion, ${percent}%`,
  caveatKicker: "Evaluation caveat",
  caveat: "Synthetic infrared is generated at 128×128, while LLVIP infrared is captured at 1280×1024. Both are subsequently placed on a canvas whose long side is 320 pixels. That shared canvas does not make their information content equivalent. The resolution mismatch may explain a substantial part of the gap between real and synthetic AP. Comparisons among the diffusion fractions use the same 128×128 generation pipeline, so they share this limit.",
  caveatUse: "Under this evaluation protocol, the generated infrared is not useful to the frozen pedestrian detector.",
  s8Title: "The hardest test frames",
  s8: "These are the test pairs with the largest generated-versus-real infrared error, ranked by mean absolute error on the 128×128 frames. The generated predictions themselves are not displayed here.",
  s8Note: "The sentences describe the real pair. They do not say what the model drew.",
  reviewMissing: "Failure review not yet written",
  scenes: {
    "190368.jpg": "Night street. A cyclist passes parked cars. In infrared the rider is bright, and the area behind the tree is bright where the color frame is dark.",
    "210143.jpg": "A person crosses near a car with its headlights on. In infrared the person is bright, and the headlight becomes a hot patch on the car.",
    "220297.jpg": "A bus fills the frame. People on the pavement and inside the windows are bright in infrared. The roof is dark in infrared and yellow under the street light in color.",
  },
  pairAlt: (id) => `Visible image and real infrared for ${id}`,
  visibleShort: "visible",
  rank: (model, worst, mae, count, total, prefix) => `${model} · worst ${worst} · MAE ${mae} · ${count} of ${total} begin with ${prefix}`,
  modelName: (baseline, percent) => (baseline ? `U-Net, ${percent}%` : `Diffusion, ${percent}%`),
  s9Title: "What this study does not contain",
  s9: "The study does not test whether synthetic infrared would help train a detector. Flow matching and video were not run. The next measurement is a resolution-matched control: downsample real infrared to 128×128, pass it through the same canvas as the generated images, and score the same frozen detector.",
  s10Title: "What this case study establishes",
  s10a: "Under a fixed LLVIP test set and a controlled nested reduction of paired training data, PSNR, SSIM, LPIPS, and pedestrian AP do not degrade in the same way. PSNR stays relatively stable, while structural and perceptual metrics deteriorate and detector performance on the generated images approaches zero.",
  s10b: "The downstream result should be read with the resolution caveat above. Synthetic infrared is generated at a much lower resolution than the original LLVIP captures. A resolution-matched control is the next experiment.",
  s11Title: "Reproducibility",
  s11: "Seed 42. Validation is cut once from the official LLVIP training list. The 1% list is a prefix of the 5% list, which is a prefix of the 20% list. Diffusion uses 30 epochs at every fraction, so the smaller lists receive fewer optimizer steps. The U-Net reference uses 20 epochs on the full training list. Scores on this page come from the result archives through scripts/export_web_results.py.",
  status: (value) => value.replaceAll("_", " "),
};

const fr: Copy = {
  langLabel: "Langue",
  kicker: "Étude de cas de recherche indépendante · 2026",
  title: "Pretty little pictures",
  lede: "L’apparence peut tromper — même une machine.",
  question: "Lorsqu’un traducteur du visible vers l’infrarouge est entraîné sur de moins en moins de paires alignées, la fidélité pixel, la similarité perceptuelle et l’utilité pour la détection de piétons se dégradent-elles ensemble ?",
  tags: ["IA générative", "Vision par ordinateur", "RVB vers infrarouge", "Apprentissage multimodal", "Rareté des données", "PyTorch"],
  measured: "Mesuré sur le jeu de test LLVIP",
  pending: "Expériences en attente",
  s1Title: "Nous voyons en RVB",
  s1: "La plupart de la perception automatique part d’une caméra ordinaire. L’image est dense, peu coûteuse, et pleine de texture. C’est la modalité que l’on sait déjà collecter.",
  s2Title: "Jusqu’à ce que le RVB ne voie plus assez",
  s2: "Sous faible lumière, la texture visible s’amincit. Une personne peut rester dans la scène alors que l’image couleur l’a perdue. La mesure thermique est une autre mesure du même lieu, pas une photographie universellement meilleure.",
  s3Title: "Même scène. Autre capteur.",
  s3Note: "Une image de test LLVIP. Faites glisser pour comparer l’image visible et l’infrarouge aligné. Le côté infrarouge est la capture réelle, pas une image générée.",
  sliderLabel: "Séparateur entre l’image visible et l’infrarouge réel",
  visible: "Visible",
  realInfrared: "Infrarouge réel",
  noPair: "Aucune paire alignée n’a encore été exportée.",
  s4Title: "Le jeu apparié est le jeu rare",
  s4Intro: "LLVIP est le jeu d’étude : des paires alignées visible et infrarouge lointain, surtout de nuit, avec des boîtes de piétons.",
  counts: (train, val, test, one) => `Les listes fixes contiennent ${train} paires d’entraînement, ${val} paires de validation et ${test} paires de test. La liste à 1 % contient ${one} paires.`,
  countsMissing: "Les effectifs restent non indiqués tant que prepare_dataset.py n’a pas lu l’archive.",
  s5Title: "Peut-on générer ce qui manque ?",
  archLabel: "Chaîne du RVB vers l’infrarouge",
  archRgb: "Image RVB",
  archModel: "Modèle conditionnel",
  archOut: "Infrarouge synthétique",
  s5: "La référence est un U-Net L1. Le modèle principal est un petit réseau de diffusion conditionnel : l’image RVB est concaténée à l’infrarouge bruité, et le réseau prédit le bruit. Il est entraîné sur les paires. Ce n’est pas un modèle de fondation photographique.",
  s6Title: "On retire maintenant les données",
  s6: "Le jeu de test reste fixe. Une seule permutation aléatoire est tirée avec la graine 42, et les listes d’entraînement plus petites sont des préfixes emboîtés de cette permutation : 1 % ⊂ 5 % ⊂ 20 % ⊂ 100 %. Le contrôle ci-dessous ne lit que l’enregistrement exporté pour cette fraction. Chaque panneau montre le RVB, puis l’infrarouge généré, puis l’infrarouge réel.",
  fractionLabel: "Fraction des données d’entraînement",
  awaiting: "Expérience en attente",
  sameList: "Même liste de test fixe pour chaque fraction.",
  noCurve: "Courbe d’entraînement pas encore disponible",
  generated: "Généré",
  panelAlt: (id) => `RVB, infrarouge généré et infrarouge réel pour ${id}`,
  chartCaption: "Diffusion seulement. Chaque panneau part de zéro et a sa propre échelle verticale. L’axe horizontal est la fraction d’entraînement.",
  hintCloser: "plus haut, plus proche",
  hintLower: "plus bas, plus proche",
  hintBetter: "plus haut, mieux",
  chartAria: (label, hint, ticks) => `${label}, ${hint}, pour les fractions d’entraînement ${ticks} pour cent`,
  valLoss: "Perte de validation",
  epoch: "Époque",
  curveLabel: "Perte de validation par époque",
  s7Title: "Joli n’est pas la même chose qu’utile",
  protocol: [
    ["Jeu de données", "LLVIP"],
    ["Paires d’entraînement", ""],
    ["Paires de validation", ""],
    ["Paires de test", ""],
    ["Fractions d’entraînement", "100 %, 20 %, 5 %, 1 %"],
    ["Découpage", "Une permutation aléatoire, graine 42 ; préfixes emboîtés"],
    ["Modèle génératif", "Diffusion conditionnelle"],
    ["Référence", "U-Net L1, données d’entraînement à 100 % seulement"],
    ["Entraînement diffusion", "30 époques par fraction"],
    ["Évaluation", "PSNR, SSIM, LPIPS, AP piéton"],
    ["Détecteur", "Détecteur de piétons gelé, entraîné sur l’infrarouge réel"],
  ],
  seedNote: "Une exécution d’entraînement par fraction, graine 42. Ces résultats sont une étude de cas indépendante, pas un banc d’essai à plusieurs graines.",
  s7Metrics: "Le PSNR et le SSIM mesurent l’accord avec l’infrarouge cible aligné. Le LPIPS mesure une distance perceptuelle. L’AP piéton pose une autre question : l’infrarouge généré reste-t-il utile à un détecteur entraîné sur l’infrarouge réel ?",
  finding: (p) =>
    `Sur les quatre fractions d’entraînement du diffusion, les métriques ne se dégradent pas ensemble. Le PSNR est resté entre ${p.min} et ${p.max} dB${p.monotonic ? "." : " et n’est pas monotone."} Le SSIM est passé de ${p.ssimFrom} à ${p.ssimTo}, tandis que le LPIPS est passé de ${p.lpipsFrom} à ${p.lpipsTo}. L’AP piéton sur les images générées est passé de ${p.apFrom} à 100 % des données à ${p.apTo} à 1 %.`,
  findingPending: "La comparaison reste non formulée tant que l’export ne contient pas tous les scores.",
  findingClose: "Le résultat n’est pas une seule notion de qualité. L’erreur pixel, la similarité perceptuelle et la détection en aval peuvent raconter des histoires différentes.",
  unetRole: "Le U-Net est une référence L1 supervisée, entraînée sur tout le jeu d’entraînement. Les ablations de rareté n’ont été faites que pour le modèle de diffusion.",
  tableNote: "Mêmes 3 463 images de test. Un PSNR, un SSIM et une AP plus hauts sont meilleurs. Un LPIPS plus bas est plus proche.",
  colImages: "Images de test",
  unetRow: "Référence U-Net, 100 %",
  diffusionRow: (percent) => `Diffusion, ${percent} %`,
  caveatKicker: "Limite d’évaluation",
  caveat: "L’infrarouge synthétique est produit en 128×128, alors que l’infrarouge LLVIP est capturé en 1280×1024. Les deux sont ensuite placés sur un canevas dont le grand côté fait 320 pixels. Ce canevas commun ne rend pas leur contenu d’information équivalent. L’écart de résolution peut expliquer une grande partie de la différence d’AP entre réel et synthétique. Les comparaisons entre fractions de diffusion utilisent le même pipeline de génération en 128×128, donc elles partagent cette limite.",
  caveatUse: "Sous ce protocole d’évaluation, l’infrarouge généré n’est pas utile au détecteur de piétons gelé.",
  s8Title: "Les images de test les plus difficiles",
  s8: "Ce sont les paires de test dont l’erreur entre infrarouge généré et infrarouge réel est la plus grande, classées par erreur absolue moyenne sur les images 128×128. Les prédictions générées ne sont pas affichées ici.",
  s8Note: "Les phrases décrivent la paire réelle. Elles ne disent pas ce que le modèle a dessiné.",
  reviewMissing: "La revue des échecs n’est pas encore écrite",
  scenes: {
    "190368.jpg": "Rue de nuit. Un cycliste passe devant des voitures garées. En infrarouge, le cycliste est clair, et la zone derrière l’arbre est claire là où l’image couleur est sombre.",
    "210143.jpg": "Une personne traverse près d’une voiture aux phares allumés. En infrarouge, la personne est claire, et le phare devient une tache chaude sur la voiture.",
    "220297.jpg": "Un bus remplit l’image. Des personnes sur le trottoir et derrière les vitres sont claires en infrarouge. Le toit est sombre en infrarouge et jaune sous l’éclairage public en couleur.",
  },
  pairAlt: (id) => `Image visible et infrarouge réel pour ${id}`,
  visibleShort: "visible",
  rank: (model, worst, mae, count, total, prefix) => `${model} · pire ${worst} · MAE ${mae} · ${count} sur ${total} commencent par ${prefix}`,
  modelName: (baseline, percent) => (baseline ? `U-Net, ${percent} %` : `Diffusion, ${percent} %`),
  s9Title: "Ce que cette étude ne contient pas",
  s9: "L’étude ne teste pas si l’infrarouge synthétique aiderait à entraîner un détecteur. Le flow matching et la vidéo n’ont pas été exécutés. La prochaine mesure est un contrôle à résolution égale : réduire l’infrarouge réel à 128×128, le passer par le même canevas que les images générées, et scorer le même détecteur gelé.",
  s10Title: "Ce que cette étude de cas établit",
  s10a: "Sur un jeu de test LLVIP fixe et une réduction emboîtée contrôlée des paires d’entraînement, le PSNR, le SSIM, le LPIPS et l’AP piéton ne se dégradent pas de la même façon. Le PSNR reste relativement stable, tandis que les métriques de structure et de perception se détériorent et que la performance du détecteur sur les images générées tend vers zéro.",
  s10b: "Le résultat en aval doit être lu avec la limite de résolution ci-dessus. L’infrarouge synthétique est généré à une résolution bien plus basse que les captures LLVIP d’origine. Un contrôle à résolution égale est la prochaine expérience.",
  s11Title: "Reproductibilité",
  s11: "Graine 42. La validation est coupée une fois dans la liste d’entraînement officielle de LLVIP. La liste à 1 % est un préfixe de la liste à 5 %, qui est un préfixe de la liste à 20 %. Le diffusion utilise 30 époques à chaque fraction, donc les listes plus petites reçoivent moins de pas d’optimisation. La référence U-Net utilise 20 époques sur toute la liste d’entraînement. Les scores de cette page viennent des archives de résultats via scripts/export_web_results.py.",
  status: (value) => ({ completed: "terminé", pending: "en attente", trained_not_evaluated: "entraîné, non évalué" }[value] ?? value.replaceAll("_", " ")),
};

export const copy: Record<Lang, Copy> = { en, fr };
