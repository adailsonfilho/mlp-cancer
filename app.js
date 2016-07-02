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
	    htmlStr = htmlStr + '<div class="panel-heading">' + '<h4>Config ' + confNum +
	            '<button type="button" class="btn btn-success mLeft10 btn-sm mBottom10 pull-right" onClick="showIt(' + confNum + ')">Show Results</button>'
	                + '</h4></div>';
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
    					} else if (key === 'topology_options') {
    					    db[key].forEach(
        						function(k) {
        							htmlStr = htmlStr + '<p>name: <code>' + k.name + 
        								'</code> units: <code>' + k.units + '</code></p>';
        						}
        					);
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
 * Show a modal with configuration curves
 * @param confNum: the configuration number
 */
function showIt(confNum) {
    var modalTemplate = function(confNum) {
        var result = '<div> ';
        
        result = result + '<img src="latest/config_' + confNum + '/Confusion matrix.png"/>';
        result = result + '<img src="latest/config_' + confNum + '/MSE Curve.png"/>';
        result = result + '<img src="latest/config_' + confNum + '/ROC Curve.png"/>';
        
        return result + '</div>';
    };
    
    // title
    $('#modal-title').text('Config ' + confNum);
    
    // content (curves)
    $('#modal-content').html(modalTemplate(confNum));
    
    // show
    $("#myModal").modal();
}

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

/**
 * Clones an object
 */
function clone(obj) {
    var copy;

    // Handle the 3 simple types, and null or undefined
    if (null == obj || "object" != typeof obj) return obj;

    // Handle Date
    if (obj instanceof Date) {
        copy = new Date();
        copy.setTime(obj.getTime());
        return copy;
    }

    // Handle Array
    if (obj instanceof Array) {
        copy = [];
        for (var i = 0, len = obj.length; i < len; i++) {
            copy[i] = clone(obj[i]);
        }
        return copy;
    }

    // Handle Object
    if (obj instanceof Object) {
        copy = {};
        for (var attr in obj) {
            if (obj.hasOwnProperty(attr)) copy[attr] = clone(obj[attr]);
        }
        return copy;
    }

    throw new Error("Unable to copy obj! Its type isn't supported.");
}

/**
 * compute responses to create data for chart
 * @param field: the field of the chart
 */
function setData(field) {
   var result = [];
   
   // random number
   var num = function(min, max) {
       if ((min !== undefined) && (max !== undefined)) {
           return Math.floor(Math.random() * (max - min + 1)) + min;
       } else {
           return num(0, 100);
       }
   };
   
   // random color (with alpha)
   var color = function(alpha) {
       return "rgba(" + num(0, 254) + "," + num(0, 254) + "," + num(0, 254) + "," + alpha + ")";
   };
   
   // colorize a template
   var colorize = function(t) {
       var c = color(1);
       
       t.borderColor = c;
       t.pointBackgroundColor = c;
       t.pointHoverBorderColor = c;
       
       t.backgroundColor = c.replace(',1)', ',0.2)');
       
       return t;
   };
   
   // template for graph
   var template = {
       label: "",
       backgroundColor: "rgba(179,181,198,0.2)",
       borderColor: "rgba(179,181,198,1)",
       pointBackgroundColor: "rgba(179,181,198,1)",
       pointBorderColor: "#fff",
       pointHoverBackgroundColor: "#fff",
       pointHoverBorderColor: "rgba(179,181,198,1)",
       data: []
   };
   
   // put responses into dataset
   Object.keys(configs.map(function(config) {
   		return config['results'];
   })).forEach(
        function(response) {
            result.push(colorize(clone(template)));
            result[result.length - 1].label = 'Config ' + response;
            
            // ["Precision", "Final MSE", "ROC Area", "Confusion"]
            result[result.length - 1].data = [
                configs[response]['results'][field]
                //configs[response]['results'].mse,
                //configs[response]['results'].roc,
                //configs[response]['results'].confusion[0][0]
            ];
    });
    return result;
}





// Rodando...
configs.map(function(obj) {
    return new Object(obj);
}).forEach(
    function(config, index) {
        appendObject('main', config, index);
    }
);

// the data of the charts
var precisionData = {
    //labels: ["Precision", "Final MSE", "ROC Area", "Confusion"],
    labels: ["Precision"],
    datasets: setData('precision')
};

var mseData = {
    //labels: ["Precision", "Final MSE", "ROC Area", "Confusion"],
    labels: ["Final MSE"],
    datasets: setData('mse')
};

var rocData = {
    //labels: ["Precision", "Final MSE", "ROC Area", "Confusion"],
    labels: ["ROC Area"],
    datasets: setData('roc')
};

// create charts
var ctx = document.getElementById("graph-precision");
new Chart(ctx, {
    type: "bar",
    data: precisionData,
    options: {
        scales: {
            reverse: false,
            ticks: {
                beginAtZero: true
            }
        }
    }
});

var ctx = document.getElementById("graph-mse");
new Chart(ctx, {
    type: "bar",
    data: mseData,
    options: {
        scales: {
            reverse: false,
            ticks: {
                beginAtZero: true
            }
        }
    }
});

var ctx = document.getElementById("graph-roc");
new Chart(ctx, {
    type: "bar",
    data: rocData,
    options: {
        scales: {
            reverse: false,
            ticks: {
                beginAtZero: true
            }
        }
    }
});