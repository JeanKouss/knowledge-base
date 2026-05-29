```
api/dataItems?fields=id,displayName,name,dimensionItemType&order=displayName:asc&paging=false
api/dataElements?fields=id,displayName,name,dimensionItemType&order=displayName:asc&paging=false&filter=dataElementGroups.id:in:[lun8UDi2IZl,Wtp4rCv8RS8]
api/indicators?fields=id,displayName,name,dimensionItemType&order=displayName:asc&paging=false&filter=indicatorGroups.id:in:[NgziHev90L2]
api/programIndicators?fields=id,displayName,name,dimensionItemType&order=displayName:asc&paging=false&filter=programIndicatorGroups.id:in:[oxjnEv4TJBa]

api/organisationUnits.json?fields=id,name,level&paging=false
```

**Get an ou childrens** :
```
api/organisationUnits.json?fields=id,name,children[id,name,parent]&filter=id:in:[yXVS0fePf7b]
# Or
ous_g = client_tg.get("/api/organisationUnits.json", params={
        "paging": False,
        "fields":"id,name,level,ancestors[id,name,level]",
        "level": 3,
        "filter": "parent.id:in:[yXVS0fePf7b]"
})
# Or
ous_g = client_tg.get("/api/organisationUnits.json", params={
        "paging": False,
        "fields":"id,name,level,ancestors[id,name,level]",
        "level": 3,
        "filter": "ancestors.id:in:[yXVS0fePf7b]"
})
```
