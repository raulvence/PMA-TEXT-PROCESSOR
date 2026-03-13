import json
import boto3
import re
import os
import requests
from requests.auth import HTTPBasicAuth

# Inicialización de clientes
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

def lambda_handler(event, context):
    try:
        # 1. Obtener datos del evento
        bucket_name = event.get('bucket')
        file_key = event.get('file_key')

        if not bucket_name or not file_key:
            return {
                "statusCode": 400,
                "error": "Faltan parámetros: bucket o file_key."
            }

        # 2. Leer transcripción desde S3
        print(f"Leyendo archivo: s3://{bucket_name}/{file_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        transcribe_data = json.loads(response['Body'].read().decode('utf-8'))

        # Extraer texto
        texto_crudo = transcribe_data['results']['transcripts'][0]['transcript']

        # 3. Prompt dinámico - Cargado desde archivo
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except FileNotFoundError:
            # Fallback si el archivo no existe
            print("Advertencia: prompt.txt no encontrado. Usando prompt por defecto.")
            system_prompt = ""


        # 6. Llamada a Bedrock
        cuerpo_peticion = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4500,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": f"Aquí está la transcripción:\n\n{texto_crudo}"}
            ]
        }

        print("Invocando a Bedrock...")
        resp = bedrock_client.invoke_model(
            modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=json.dumps(cuerpo_peticion)
        )

        resultado_claude = json.loads(resp['body'].read())['content'][0]['text']
        
        # Log para ver la respuesta cruda en CloudWatch
        print("=== RESPUESTA CRUDA DE CLAUDE ===")
        print(resultado_claude)

        # 7. Parseo robusto
        def extraer_seccion(tag, texto):
            pattern = rf"\[{tag}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)"
            match = re.search(pattern, texto, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        titulo = extraer_seccion("TITULO_H1", resultado_claude)
        slug = extraer_seccion("URL_SLUG", resultado_claude)
        meta_desc = extraer_seccion("META_DESCRIPCION", resultado_claude)
        keywords = extraer_seccion("PALABRAS_CLAVE", resultado_claude)
        cuerpo_html = extraer_seccion("CUERPO_HTML", resultado_claude)
        image_prompt = extraer_seccion("IMAGE_PROMPT", resultado_claude)

        if not cuerpo_html:
            cuerpo_html = resultado_claude

        # 8. Validaciones SEO

        # Normalizar slug para evitar errores 500 en WordPress
        slug = slug.lower()
        slug = re.sub(r'[^a-z0-9\-]', '', slug)
        slug = re.sub(r'-+', '-', slug).strip('-')

        # Si por alguna razón extrema la IA no generó slug, armamos uno seguro basado en el título
        if not slug and titulo:
            slug = re.sub(r'\W+', '-', titulo.lower()).strip('-')

        # --- LIMPIEZA DE CARACTERES DE ESCAPE PARA HTML ---
        # Convertimos los saltos de línea literales en espacios, ya que las etiquetas HTML manejan el espaciado
        cuerpo_html = cuerpo_html.replace('\\n', ' ')
        # Limpiamos comillas escapadas si las hubiera
        cuerpo_html = cuerpo_html.replace('\\"', '"')
        cuerpo_html = cuerpo_html.replace('«', '"').replace('»', '"')
        # Limpiamos dobles espacios que se hayan generado al unir
        cuerpo_html = re.sub(r'\s+', ' ', cuerpo_html).strip()

        # --- FUNCION PARA OBTENER O CREAR TAGS EN WORDPRESS ---

        def obtener_id_del_tag(nombre_tag, wp_url, auth):
            try:
                # Buscar si ya existe
                search_url = f"{wp_url}/wp-json/wp/v2/tags"
                res = requests.get(
                    search_url,
                    params={"search": nombre_tag},
                    auth=auth,
                    timeout=15
                )

                if res.status_code == 200:
                    for t in res.json():
                        if t['name'].lower() == nombre_tag.lower():
                            return t['id']

                # Si no existe, crearlo
                create_res = requests.post(
                    search_url,
                    json={"name": nombre_tag},
                    auth=auth,
                    timeout=15
                )

                if create_res.status_code in [200, 201]:
                    return create_res.json().get('id')

                print(f"No se pudo crear tag: {nombre_tag}")
                return None

            except Exception as e:
                print(f"Error procesando tag '{nombre_tag}': {str(e)}")
                return None

        # --- PUBLICAR EN WORDPRESS ---

        wp_url = os.environ.get("WP_URL")
        wp_user = os.environ.get("WP_USER")
        wp_app_password = os.environ.get("WP_APP_PASSWORD")
        post_id = event.get("post_id")

        if wp_url and wp_user and wp_app_password and post_id:

            auth = HTTPBasicAuth(wp_user, wp_app_password)

            # Convertir keywords string a lista limpia
            lista_keywords = [
                k.strip() for k in keywords.split(",") if k.strip()
            ]

            # Obtener IDs reales de WordPress
            lista_ids_tags = [
                obtener_id_del_tag(k, wp_url, auth)
                for k in lista_keywords
            ]

            # Limpiar posibles None
            lista_ids_tags = [tid for tid in lista_ids_tags if tid]

            endpoint = f"{wp_url}/wp-json/wp/v2/posts/{post_id}"

            payload_wp = {
                "title": titulo,
                "content": cuerpo_html,
                "excerpt": meta_desc,
                "slug": slug,
                "status": "publish",
                "tags": lista_ids_tags,
                "acf": {
                    "video_transcript": texto_crudo
                }
            }

            print("Enviando actualización a WordPress...")

            response_wp = requests.post(
                endpoint,
                json=payload_wp,
                auth=auth,
                timeout=90
            )

            print("Status WP:", response_wp.status_code)
            print("Respuesta WP:", response_wp.text)

        else:
            print("No se ejecutó publicación WP: faltan variables o post_id")    

        # 9. Retorno final
        return {
            "post_id": post_id,
            "image_prompt": image_prompt,
            "post_title": titulo,
            "post_slug": slug
        }

    except Exception as e:
        print(f"Error en ejecución: {str(e)}")
        raise e