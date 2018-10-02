canvas = {
    'init_script': 'document.getElementsByClassName("runner-canvas")[0].id = '
                   '"runner-canvas"',
    'get_base64_script': "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"
}
