// Regression coverage for the shipped page's bridge contract, not native UI acceptance.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync(require('node:path').join(__dirname,'../src/editor.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function page(bridge){
 const events={}, messages=[];
 const status={};
 const buttons=['input','output','close'].map(action=>({dataset:{action},addEventListener(_,f){this.click=f}}));
 const nodes={editor:{addEventListener(n,f){events[n]=f},reportValidity(){return true}},status,input:{},output:{}};
 const document={getElementById:id=>nodes[id],querySelectorAll:q=>q==='[data-action]'?buttons:buttons};
 const window=bridge?{chrome:{webview:{postMessage(m){if(bridge==='throw')throw Error('closed');messages.push(m)},addEventListener(n,f){events[n]=f}}}}:{};
 vm.runInNewContext(script,{document,window,FormData:class{*[Symbol.iterator](){yield ['patient_id','SYNTHETIC']}}});
 return {buttons,nodes,events,messages};
}
let p=page(false);
assert.match(p.nodes.status.textContent,/Browser preview only/);
p.buttons[0].click(); assert.match(p.nodes.status.textContent,/requires ritk-anonymizer.exe/);
p=page(true); p.buttons[0].click(); assert.equal(p.messages[0].action,'input');
p.buttons[1].click(); assert.equal(p.messages[1].action,'output');
p.events.message({data:{kind:'input',message:'C:\\Synthetic'}});assert.equal(p.nodes.input.textContent,'C:\\Synthetic');
p.events.message({data:{kind:'busy',message:'Working'}});p.buttons[0].click();assert.equal(p.messages.length,2);
p.events.message({data:{kind:'error',message:'Picker failed'}});p.buttons[0].click();assert.equal(p.messages.length,3);
p=page('throw');p.buttons[0].click();assert.match(p.nodes.status.textContent,/Unable to contact/);
console.log('editor bridge regression tests passed');
