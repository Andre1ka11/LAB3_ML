"""Figures for GigaChat and the available paired three-model experiment."""
import os
import numpy as np
from analyze_yandex import read_json,read_lines,accuracy_ci


def plot(root,out,types,yandex,classification,hypotheses,costs):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'figures';dest.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=150);plt.close()
    qwen=read_json(out/'comparison_scores_qwen.json')
    type_rows=lambda rs:[{'type':t,**accuracy_ci([r['correct'] for r in rs if r['type']==t])} for t in 'ABF']
    models={'YandexGPT':type_rows(yandex),'Qwen3-4B':type_rows(qwen),'GigaChat-2':types}
    fig,ax=plt.subplots(figsize=(7,4))
    for offset,(name,rows) in zip([-.12,0,.12],models.items()):
        ax.errorbar(np.arange(3)+offset,[r['accuracy'] for r in rows],yerr=[[r['accuracy']-r['ci_low'] for r in rows],[r['ci_high']-r['accuracy'] for r in rows]],fmt='o',capsize=4,label=name)
    ax.set(xticks=range(3),xticklabels=list('ABF'),ylim=(-.05,1.05),ylabel='Accuracy',title='95% bootstrap CI, n=8/type');ax.legend();save('accuracy_ci')
    fig,ax=plt.subplots(figsize=(7,3.4));heat=np.array([[next((r['accuracy'] for r in rows if r['type']==t),np.nan) for t in 'ABCDF'] for rows in models.values()])
    cmap=plt.get_cmap('YlGn').copy();cmap.set_bad('lightgray');im=ax.imshow(heat,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(5),xticklabels=list('ABCDF'),yticks=range(3),yticklabels=list(models))
    for (i,j),v in np.ndenumerate(heat):ax.text(j,i,'pending' if np.isnan(v) else f'{v:.1%}',ha='center',va='center')
    fig.colorbar(im,ax=ax);save('heatmap')
    fig,ax=plt.subplots(figsize=(7,4))
    for model in ['GigaChat-2','LogisticRegression','GaussianNB']:
        rows=[r for r in classification if r['model']==model];ax.plot([r['shots'] for r in rows],[r['f1_minority'] for r in rows],'o-',label=model)
    ax.set(xlabel='Shots',ylabel='F1 class 1',ylim=(-.05,1.05));ax.legend();save('classification_f1')
    fig,ax=plt.subplots(figsize=(5,4));h=hypotheses['H10'];bins=[b for b in h['bins'] if b['n']]
    ax.plot([0,1],[0,1],'--',color='gray');ax.scatter([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],s=[b['n']*15 for b in bins])
    ax.set(xlim=(0,1.03),ylim=(0,1.03),xlabel='Mean confidence',ylabel='Accuracy',title=f'H10: {h["n_valid"]}/24 valid');save('reliability')
    fig,ax=plt.subplots(figsize=(6,4))
    for r in hypotheses['H12']['comparisons']:
        d=r['delta_f1_llm_minus_lr'];ax.errorbar([r['shots']],[d],yerr=[[d-r['one_sided_95_lower']],[r['one_sided_95_upper']-d]],fmt='o',capsize=4)
    ax.axhline(0,color='gray');ax.axhline(-.05,color='red',ls='--');ax.set(xticks=[5,20],xlabel='Shots',ylabel='F1 GigaChat - F1 LR',title='H12: one-sided 95% bounds');save('h12_delta')
    fig,ax=plt.subplots(figsize=(7,4))
    for i,h in enumerate([hypotheses['H5'],hypotheses['H5']['secondary_gigachat_vs_qwen']]):
        d=h['statistic_gap_A_minus_gap_B'];ax.errorbar([i],[d],yerr=[[d-h['ci_low']],[h['ci_high']-d]],fmt='o',capsize=4)
    ax.axhline(0,color='gray');ax.set(xticks=[0,1],xticklabels=['Primary: Yandex vs Qwen','Secondary: GigaChat vs Qwen'],ylabel='Gap A minus gap B',title='H5: paired bootstrap');save('h5_gap')
    fig,ax=plt.subplots(figsize=(7,3));ax.bar([r['stage'] for r in costs],[r['mean_latency_s'] for r in costs]);ax.set(ylabel='Mean latency (s)',title='GigaChat-2 API latency');save('runtime')
    fig,ax=plt.subplots(figsize=(7,4))
    for name,path in [('YandexGPT','results/models/yandex/yandex-v1/responses.jsonl'),('Qwen3-4B','results/models/local/qwen3-4b-q4-v1/responses.jsonl'),('GigaChat-2','results/models/gigachat/gigachat-2-v1/responses.jsonl')]:
        rs=[r for r in read_lines(root/path) if r['stage']=='main' and r['task_id'][0] in 'ABF'];cost=sum(r['cost_rub'] for r in rs)
        acc=np.mean([r['accuracy'] for r in models[name]])
        ax.scatter([cost],[acc]);ax.annotate(name,(cost,acc),xytext=(-8,8) if name=='YandexGPT' else (8,8),
            ha='right' if name=='YandexGPT' else 'left',textcoords='offset points')
    ax.set(xlabel='API tariff-equivalent cost A+B+F (RUB)',ylabel='Accuracy',ylim=(0,1.08),title='Freemium spending and local hardware costs excluded');ax.margins(x=.2);save('cost_accuracy')
