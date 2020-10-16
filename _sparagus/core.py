ANALYTICS_URL = 'https://sparagus.pythonanywhere.com/ping'


def make_analytics_snippet():
    return f'''
<script>
    const path = window.location.pathname;
    const url = '{ANALYTICS_URL}?path=' + window.location.pathname;
    fetch(url, {{ mode: 'no-cors', cache: 'no-cache' }});
</script>
    '''
