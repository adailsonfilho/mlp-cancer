/**
 * Usado para formatar números na matriz de confusão
 * @param pad: o formato ta string de dígitos (em zeros)
 */
Number.prototype.toDigits = function(pad) {
    var n = '' + this;
    var ans = pad.substring(0, pad.length - n.length) + n;
    return ans;
};

/**
 * Geração de HTML da config aqui
 * @param cls: classe a ser colocada na div
 * @param confNum: the number of config in array
 */
Object.prototype.toHtml = function(cls='none', confNum) {
	var db = this;
	var htmlStr = '';
	if (cls !== 'none') {
	    htmlStr = '<div id="' + 'config_' + confNum + '" class="panel panel-default ' + (cls || '') + '">';
	    htmlStr = htmlStr + '<div class="panel-heading">' + 'Config ' + confNum + '</div>';
	    htmlStr = htmlStr + '<div class="panel-body">';
	}

	Object.keys(db).forEach(
		function(key) {
			switch (typeof db[key]) {
				case 'string':
				    htmlStr = htmlStr + '<p>'
    						+ key + ': <code>' + db[key] + '</code></p>';
					break;
				case 'number':
					htmlStr = htmlStr + '<p>'
						+ key + ': <code>' + db[key] + '</code></p>';
					break;
				case 'object':
					if (db[key] instanceof Array) {
						htmlStr = htmlStr + '<p>' + key + ':<ul>';
    					if (key === 'confusion') {
    					    htmlStr = htmlStr + '<p><code>| ' + db[key][0][0].toDigits('0000')
    					        + ' ' + db[key][0][1].toDigits('0000') + ' |</code></p>';
						    
						    htmlStr = htmlStr + '<p><code>| ' + db[key][1][0].toDigits('0000')
    					        + ' ' + db[key][1][1].toDigits('0000') + ' |</code></p>';
    					} else {
    					    db[key].forEach(
        						function(k) {
        							htmlStr = htmlStr +
        								new Object(k.toHtml('none', confNum));
        						}
        					);
    					}
    					htmlStr = htmlStr + '</ul>';
					} else {
						htmlStr = htmlStr + '<p>' + key + ':</p>' +
						    '<div class="col-xs-11 col-xs-offset-1">'
								+ new Object(db[key]).toHtml('none', confNum) + '</div>';
					}
					break;
			}
		}
	);
	
	if (cls !== 'none') {
	    htmlStr = htmlStr + '</div>';
	}
	
	return htmlStr;
};

/**
 * Responsável por 'pendurar'dados no HTML
 * @param id: the id of HTML element
 * @param obj: the object to append
 * @param index: position of object
 */
function appendObject(id, obj, index) {
	var div, target = document.getElementById(id);
	div = document.createElement('div');
	
	div.innerHTML = obj.toHtml('config', index);
	target.appendChild(div);
	
	while (div.firstChild) {
        // Also removes child nodes from 'div'
        target.insertBefore(div.firstChild, div);
    }
    // Remove 'div' element from target element
    target.removeChild(div);
}





// Rodando...
configs.map(function(obj) {
    return new Object(obj);
}).forEach(
    function(config, index) {
        appendObject('main', config, index);
    }
);