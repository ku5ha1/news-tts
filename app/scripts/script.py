# import asyncio
# from motor.motor_asyncio import AsyncIOMotorClient

# async def migrate_storage_urls():
#     # MongoDB connection
#     MONGO_URI = "mongodb+srv://sushanth:mar5h8UdgeCNgPOP@cluster0.aabkw.mongodb.net/test?retryWrites=true&w=majority&appName=Cluster0"
#     client = AsyncIOMotorClient(MONGO_URI)
#     db = client["test"]
    
#     old_domain = "diprstorage.blob.core.windows.net"
#     new_domain = "diprstorageindia.blob.core.windows.net"
    
#     print("Starting MongoDB URL migration...")
#     print(f"Old domain: {old_domain}")
#     print(f"New domain: {new_domain}\n")
    
#     # 1. Update news collection
#     print("Updating news collection...")
#     result = await db.news.update_many(
#         {"$or": [
#             {"hindi.audio_description": {"$regex": f"^https://{old_domain}"}},
#             {"kannada.audio_description": {"$regex": f"^https://{old_domain}"}},
#             {"English.audio_description": {"$regex": f"^https://{old_domain}"}},
#             {"newsImage": {"$regex": f"^https://{old_domain}"}}
#         ]},
#         [{"$set": {
#             "hindi.audio_description": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$hindi.audio_description", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$hindi.audio_description", "find": old_domain, "replacement": new_domain}},
#                     "else": "$hindi.audio_description"
#                 }
#             },
#             "kannada.audio_description": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$kannada.audio_description", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$kannada.audio_description", "find": old_domain, "replacement": new_domain}},
#                     "else": "$kannada.audio_description"
#                 }
#             },
#             "English.audio_description": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$English.audio_description", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$English.audio_description", "find": old_domain, "replacement": new_domain}},
#                     "else": "$English.audio_description"
#                 }
#             },
#             "newsImage": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$newsImage", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$newsImage", "find": old_domain, "replacement": new_domain}},
#                     "else": "$newsImage"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ News: {result.modified_count} documents updated")
    
#     # 2. Update magazines collection
#     print("\nUpdating magazines collection...")
#     result = await db.magazines.update_many(
#         {"$or": [
#             {"magazinePdf": {"$regex": f"^https://{old_domain}"}},
#             {"magazineThumbnail": {"$regex": f"^https://{old_domain}"}}
#         ]},
#         [{"$set": {
#             "magazinePdf": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$magazinePdf", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$magazinePdf", "find": old_domain, "replacement": new_domain}},
#                     "else": "$magazinePdf"
#                 }
#             },
#             "magazineThumbnail": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$magazineThumbnail", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$magazineThumbnail", "find": old_domain, "replacement": new_domain}},
#                     "else": "$magazineThumbnail"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Magazines: {result.modified_count} documents updated")
    
#     # 3. Update magazine2 collection
#     print("\nUpdating magazine2 collection...")
#     result = await db.magazine2.update_many(
#         {"$or": [
#             {"magazinePdf": {"$regex": f"^https://{old_domain}"}},
#             {"magazineThumbnail": {"$regex": f"^https://{old_domain}"}}
#         ]},
#         [{"$set": {
#             "magazinePdf": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$magazinePdf", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$magazinePdf", "find": old_domain, "replacement": new_domain}},
#                     "else": "$magazinePdf"
#                 }
#             },
#             "magazineThumbnail": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$magazineThumbnail", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$magazineThumbnail", "find": old_domain, "replacement": new_domain}},
#                     "else": "$magazineThumbnail"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Magazine2: {result.modified_count} documents updated")
    
#     # 4. Update longvideos collection
#     print("\nUpdating longvideos collection...")
#     result = await db.longvideos.update_many(
#         {"$or": [
#             {"thumbnail": {"$regex": f"^https://{old_domain}"}},
#             {"video_url": {"$regex": f"^https://{old_domain}"}}
#         ]},
#         [{"$set": {
#             "thumbnail": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$thumbnail", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$thumbnail", "find": old_domain, "replacement": new_domain}},
#                     "else": "$thumbnail"
#                 }
#             },
#             "video_url": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$video_url", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$video_url", "find": old_domain, "replacement": new_domain}},
#                     "else": "$video_url"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Long videos: {result.modified_count} documents updated")
    
#     # 5. Update shortvideos collection
#     print("\nUpdating shortvideos collection...")
#     result = await db.shortvideos.update_many(
#         {"$or": [
#             {"thumbnail": {"$regex": f"^https://{old_domain}"}},
#             {"video_url": {"$regex": f"^https://{old_domain}"}}
#         ]},
#         [{"$set": {
#             "thumbnail": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$thumbnail", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$thumbnail", "find": old_domain, "replacement": new_domain}},
#                     "else": "$thumbnail"
#                 }
#             },
#             "video_url": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$video_url", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$video_url", "find": old_domain, "replacement": new_domain}},
#                     "else": "$video_url"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Short videos: {result.modified_count} documents updated")
    
#     # 6. Update photos collection
#     print("\nUpdating photos collection...")
#     result = await db.photos.update_many(
#         {"photoImage": {"$regex": f"^https://{old_domain}"}},
#         [{"$set": {
#             "photoImage": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$photoImage", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$photoImage", "find": old_domain, "replacement": new_domain}},
#                     "else": "$photoImage"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Photos: {result.modified_count} documents updated")
    
#     # 7. Update staticpages collection
#     print("\nUpdating staticpages collection...")
#     result = await db.staticpages.update_many(
#         {"staticpageImage": {"$regex": f"^https://{old_domain}"}},
#         [{"$set": {
#             "staticpageImage": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$staticpageImage", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$staticpageImage", "find": old_domain, "replacement": new_domain}},
#                     "else": "$staticpageImage"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Static pages: {result.modified_count} documents updated")
    
#     # 8. Update users collection (profile images)
#     print("\nUpdating users collection...")
#     result = await db.users.update_many(
#         {"profileImage": {"$regex": f"^https://{old_domain}"}},
#         [{"$set": {
#             "profileImage": {
#                 "$cond": {
#                     "if": {"$regexMatch": {"input": "$profileImage", "regex": old_domain}},
#                     "then": {"$replaceAll": {"input": "$profileImage", "find": old_domain, "replacement": new_domain}},
#                     "else": "$profileImage"
#                 }
#             }
#         }}]
#     )
#     print(f"✅ Users: {result.modified_count} documents updated")
    
#     print("\n" + "="*50)
#     print("✅ MongoDB URL migration completed successfully!")
#     print("="*50)
    
#     client.close()

# if __name__ == "__main__":
#     asyncio.run(migrate_storage_urls())

import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

def migrate_search_index_urls():
    # Azure Search credentials
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    
    if not all([search_endpoint, search_key, index_name]):
        print("❌ Missing Azure Search credentials in .env")
        return
    
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )
    
    old_domain = "diprstorage.blob.core.windows.net"
    new_domain = "diprstorageindia.blob.core.windows.net"
    
    print("Starting Azure Search index URL migration...")
    print(f"Index: {index_name}")
    print(f"Old domain: {old_domain}")
    print(f"New domain: {new_domain}\n")
    
    try:
        # Get all documents from search index
        results = search_client.search(
            search_text="*",
            select=["id", "pdf_url", "thumbnail_url"],
            include_total_count=True
        )
        
        documents_to_update = []
        total_count = 0
        
        for result in results:
            total_count += 1
            doc = {
                "id": result["id"]
            }
            updated = False
            
            # Update pdf_url if it contains old domain
            if "pdf_url" in result and result["pdf_url"] and old_domain in result["pdf_url"]:
                doc["pdf_url"] = result["pdf_url"].replace(old_domain, new_domain)
                updated = True
                print(f"  Updating pdf_url: {result['id']}")
            
            # Update thumbnail_url if it contains old domain
            if "thumbnail_url" in result and result["thumbnail_url"] and old_domain in result["thumbnail_url"]:
                doc["thumbnail_url"] = result["thumbnail_url"].replace(old_domain, new_domain)
                updated = True
                print(f"  Updating thumbnail_url: {result['id']}")
            
            if updated:
                documents_to_update.append(doc)
        
        print(f"\nTotal documents in index: {total_count}")
        print(f"Documents to update: {len(documents_to_update)}")
        
        # Update documents in batches of 100
        if documents_to_update:
            batch_size = 100
            for i in range(0, len(documents_to_update), batch_size):
                batch = documents_to_update[i:i + batch_size]
                result = search_client.merge_documents(documents=batch)
                print(f"✅ Updated batch {i//batch_size + 1}: {len(batch)} documents")
            
            print(f"\n✅ Successfully updated {len(documents_to_update)} search index documents")
        else:
            print("\n✅ No documents needed updating")
        
    except Exception as e:
        print(f"❌ Error updating search index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    migrate_search_index_urls()
