import os
import json
import threading
from queue import Queue, Empty
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

from trainer import Trainer
from features import extract_features_from_df, FEATURE_COUNT
from detect_cancer import ClinVarCancerDetector

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, 'clinvar_extracted.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_clinvar')
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_queue = Queue()
trainer = Trainer(log_queue)
detector = None

def get_detector():
    global detector
    if detector is None:
        model_path = find_latest_model()
        if model_path:
            try:
                detector = ClinVarCancerDetector(model_path=model_path)
            except Exception as e:
                print(f"Could not load detector: {e}")
    return detector

def find_latest_model():
    if os.path.exists(OUTPUT_DIR):
        pt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')]
        if pt_files:
            latest = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)))
            return os.path.join(OUTPUT_DIR, latest)
    alt_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'output_clinvar')
    if os.path.exists(alt_path):
        alt_files = [f for f in os.listdir(alt_path) if f.endswith('.pt')]
        if alt_files:
            latest = max(alt_files, key=lambda f: os.path.getmtime(os.path.join(alt_path, f)))
            return os.path.join(alt_path, latest)
    return None


@app.route('/')
def index():
    model_path = find_latest_model()
    model_available = model_path is not None and os.path.exists(model_path)
    data_available = os.path.exists(DEFAULT_DATA_PATH)
    model_timestamp = None
    if model_available:
        model_timestamp = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M')
    return render_template('index.html',
                           model_available=model_available,
                           data_available=data_available,
                           trainer_busy=trainer.is_training,
                           model_timestamp=model_timestamp)


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        if trainer.is_training:
            return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400

        data_path = request.form.get('data_path', DEFAULT_DATA_PATH)
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': f'Data file not found: {data_path}'}), 400

        config = {
            'data_path': data_path,
            'epochs': int(request.form.get('epochs', 5)),
            'batch_size': int(request.form.get('batch_size', 64)),
            'learning_rate': float(request.form.get('learning_rate', 0.001)),
            'dropout': float(request.form.get('dropout', 0.3)),
            'validation_split': float(request.form.get('validation_split', 0.2)),
            'hidden_dims': [int(x.strip()) for x in request.form.get('hidden_dims', '128,64,32').split(',')],
            'scheduler_type': request.form.get('scheduler_type', 'ReduceLROnPlateau'),
            'gradient_clip': float(request.form.get('gradient_clip', 1.0)),
            'use_class_weights': request.form.get('use_class_weights') == 'true',
            'random_seed': int(request.form.get('random_seed', 42)),
            'sample_size': int(request.form.get('sample_size', 50000)),
        }

        def run_training():
            global detector
            trainer.train(**config)
            detector = None

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        return jsonify({'status': 'ok', 'message': 'Training started'})

    return render_template('train.html',
                           data_path=DEFAULT_DATA_PATH,
                           data_exists=os.path.exists(DEFAULT_DATA_PATH),
                           trainer_busy=trainer.is_training)


@app.route('/train/stream')
def train_stream():
    def generate():
        # Send state snapshot for new connections
        if hasattr(trainer, 'get_state'):
            state = trainer.get_state()
            if state['is_training'] or state['log_buffer']:
                yield f"data: {json.dumps({'type': 'snapshot', 'state': state})}\n\n"
        while True:
            try:
                msg = log_queue.get(timeout=1)
                if msg.get('type') == 'complete' and 'chart_files' in msg.get('metrics', {}):
                    chart_urls = {}
                    for key, fname in msg['metrics']['chart_files'].items():
                        chart_urls[key] = f'/charts/{fname}'
                    msg['metrics']['chart_urls'] = chart_urls
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get('type') in ('complete', 'error'):
                    break
            except Empty:
                if not trainer.is_training:
                    break
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    return Response(generate(), mimetype='text/event-stream')


@app.route('/train/stop', methods=['POST'])
def train_stop():
    trainer.stop_requested = True
    return jsonify({'status': 'ok', 'message': 'Stop requested'})


@app.route('/train/status')
def train_status():
    return jsonify({
        'is_training': trainer.is_training,
    })


@app.route('/train/state')
def train_state():
    state = trainer.get_state() if hasattr(trainer, 'get_state') else {'is_training': False}
    return jsonify(state)


@app.route('/charts/<path:filename>')
def serve_chart(filename):
    chart_dir = os.path.join(OUTPUT_DIR, 'charts')
    return send_from_directory(chart_dir, filename)


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    det = get_detector()
    model_available = det is not None

    if request.method == 'POST' and model_available:
        try:
            form_data = {
                'chromosome': request.form.get('chromosome', '1'),
                'position': request.form.get('position', '0'),
                'gene': request.form.get('gene', 'UNKNOWN'),
                'variant_type': request.form.get('variant_type', 'SNP'),
                'ref_allele': request.form.get('ref_allele', ''),
                'alt_allele': request.form.get('alt_allele', ''),
                'clinical_significance': request.form.get('clinical_significance', ''),
            }
            result = det.detect(
                chromosome=form_data['chromosome'],
                position=int(form_data['position']),
                gene=form_data['gene'],
                variant_type=form_data['variant_type'],
                ref_allele=form_data['ref_allele'],
                alt_allele=form_data['alt_allele'],
                variation_id=int(request.form.get('variation_id', 0)),
                chr_grch38=int(request.form.get('chr_grch38', 0)),
                start_grch38=int(request.form.get('start_grch38', 0)),
                stop_grch38=int(request.form.get('stop_grch38', 0)),
                clinical_significance=form_data['clinical_significance'],
            )
            return render_template('detect.html', result=result, model_available=True, form_data=form_data)
        except Exception as e:
            return render_template('detect.html', error=str(e), model_available=True)

    return render_template('detect.html', model_available=model_available)


@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    det = get_detector()
    if det is None:
        return jsonify({'status': 'error', 'message': 'No trained model available'}), 400

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Please upload a CSV file'}), 400

    upload_path = os.path.join(SCRIPT_DIR, 'upload_' + file.filename)
    file.save(upload_path)

    output_path = os.path.join(SCRIPT_DIR, f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    try:
        results_df = det.detect_from_csv(upload_path, output_path)
        csv_data = results_df.to_csv(index=False)
        os.remove(upload_path)
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=detection_results.csv'}
        )
    except Exception as e:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def api_detect():
    det = get_detector()
    if det is None:
        return jsonify({'error': 'No trained model available'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    try:
        result = det.detect(
            chromosome=data.get('chromosome', '1'),
            position=int(data.get('position', 0)),
            gene=data.get('gene', 'UNKNOWN'),
            variant_type=data.get('variant_type', 'SNP'),
            ref_allele=data.get('ref_allele', ''),
            alt_allele=data.get('alt_allele', ''),
            variation_id=int(data.get('variation_id', 0)),
            chr_grch38=int(data.get('chr_grch38', 0)),
            start_grch38=int(data.get('start_grch38', 0)),
            stop_grch38=int(data.get('stop_grch38', 0)),
            clinical_significance=data.get('clinical_significance', ''),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    if not os.path.exists(DEFAULT_DATA_PATH):
        return render_template('stats.html', data_exists=False, stats={})

    try:
        from features import REQUIRED_COLUMNS
        sample_n = 100000
        df = pd.read_csv(DEFAULT_DATA_PATH, usecols=REQUIRED_COLUMNS, nrows=sample_n, dtype={'CHR_GRCh38': str}, low_memory=False)
        total_est = 4526899
        sig_series = df['Clinical_Significance'].astype(str).str.lower()
        pathogenic = sig_series.str.contains('pathogenic|cancer', regex=True, na=False).sum()
        total_cols = len(REQUIRED_COLUMNS)
        stats_data = {
            'total_rows': f"{total_est:,}",
            'sampled': f"{sample_n:,} rows used for preview",
            'total_columns': f"{total_cols} (used) / 299 (total in CSV)",
            'pathogenic': f"{pathogenic:,}",
            'benign': f"{len(df) - pathogenic:,}",
            'pathogenic_pct': f"{100*pathogenic/len(df):.1f}%",
            'numeric_features': ['VariationID', 'CHR_GRCh38', 'Start_GRCh38', 'Stop_GRCh38'],
            'categorical_features': ['VariationType', 'GeneSymbol'],
            'feature_count': FEATURE_COUNT,
            'memory_gb': 'N/A (streaming)',
            'memory_note': 'Reads CSV in streaming mode; no full load needed',
        }
    except Exception as e:
        stats_data = {'error': str(e)}
    return render_template('stats.html', data_exists=True, stats=stats_data)


@app.route('/models')
def models_list():
    pt_files = []
    if os.path.exists(OUTPUT_DIR):
        files = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')],
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
            reverse=True
        )
        for f in files:
            fpath = os.path.join(OUTPUT_DIR, f)
            config_basename = f.replace('clinvar_model_', 'config_').replace('.pt', '.json')
            if config_basename == f.replace('.pt', '.json'):
                config_basename = 'config_' + config_basename
            config_path = os.path.join(OUTPUT_DIR, config_basename)
            config = {}
            if os.path.exists(config_path):
                with open(config_path) as fp:
                    config = json.load(fp)
            size_mb = os.path.getsize(fpath) / 1024**2
            pt_files.append({
                'filename': f,
                'mtime': datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S'),
                'size_mb': f'{size_mb:.1f}',
                'config': config,
                'is_active': fpath == find_latest_model(),
            })
    return render_template('models.html', models=pt_files)


@app.route('/api/models')
def api_models():
    models_list_data = []
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.pt'):
                fpath = os.path.join(OUTPUT_DIR, f)
                config_basename = f.replace('clinvar_model_', 'config_').replace('.pt', '.json')
                if config_basename == f.replace('.pt', '.json'):
                    config_basename = 'config_' + config_basename
                config_path = os.path.join(OUTPUT_DIR, config_basename)
                config = {}
                if os.path.exists(config_path):
                    with open(config_path) as fp:
                        config = json.load(fp)
                models_list_data.append({
                    'filename': f,
                    'mtime': datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
                    'size_bytes': os.path.getsize(fpath),
                    'config': config,
                })
    return jsonify(models_list_data)


@app.route('/api/docs')
def api_docs():
    return render_template('api_docs.html')


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route('/charts/gallery')
def chart_gallery():
    chart_dir = os.path.join(OUTPUT_DIR, 'charts')
    chart_files = []
    if os.path.exists(chart_dir):
        files = sorted(os.listdir(chart_dir), key=lambda f: os.path.getmtime(os.path.join(chart_dir, f)), reverse=True)
        for f in files:
            if f.endswith('.png'):
                chart_files.append({
                    'filename': f,
                    'url': f'/charts/{f}',
                    'mtime': datetime.fromtimestamp(os.path.getmtime(os.path.join(chart_dir, f))).strftime('%Y-%m-%d %H:%M:%S'),
                })
    return render_template('chart_gallery.html', charts=chart_files)


@app.route('/train/config')
def train_config():
    return jsonify({
        'data_path': DEFAULT_DATA_PATH,
        'data_exists': os.path.exists(DEFAULT_DATA_PATH),
        'epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.001,
        'dropout': 0.3,
        'validation_split': 0.2,
        'hidden_dims': '128,64,32',
        'scheduler_type': 'ReduceLROnPlateau',
        'gradient_clip': 1.0,
        'use_class_weights': True,
        'random_seed': 42,
        'sample_size': 50000,
        'feature_count': FEATURE_COUNT,
        'required_columns': [
            'VariationID', 'CHR_GRCh38', 'Start_GRCh38', 'Stop_GRCh38',
            'VariationType', 'GeneSymbol', 'Ref_Allele', 'Alt_Allele',
            'Clinical_Significance'
        ],
    })


if __name__ == '__main__':
    print("=" * 60)
    print("ClinVar Cancer Detection - Flask Web App")
    print("=" * 60)
    print(f"Data path: {DEFAULT_DATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Data exists: {os.path.exists(DEFAULT_DATA_PATH)}")
    print(f"Model exists: {find_latest_model() is not None}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
